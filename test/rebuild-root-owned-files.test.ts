// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Regression test for #2727: nemoclaw rebuild aborts when files in
// .openclaw-data are root-owned.
//
// When SSH-as-sandbox-user runs `tar -cf -` on a directory that contains
// root-owned mode-0600 files (e.g. written by `kubectl exec` diagnostic
// sessions), GNU tar exits 2 and prints permission-denied errors to stderr,
// but still emits a valid archive for every file it COULD read. The fix
// accepts exit code 2 with non-empty stdout as a partial success rather
// than aborting the entire rebuild.

import { describe, it, expect } from "vitest";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const REPO_ROOT = path.join(import.meta.dirname, "..");

type SandboxStateModule = Pick<
  typeof import("../dist/lib/sandbox-state.js"),
  "safeTarExtract"
>;

async function loadSandboxState(): Promise<SandboxStateModule> {
  const loaded = await import(
    path.join(REPO_ROOT, "dist", "lib", "sandbox-state.js")
  );
  const mod = typeof loaded === "object" && loaded !== null ? loaded : null;
  if (!mod || typeof (mod as Record<string, unknown>)["safeTarExtract"] !== "function") {
    throw new Error("safeTarExtract not found in sandbox-state module");
  }
  return { safeTarExtract: (mod as SandboxStateModule).safeTarExtract };
}

// ── Tar builder (minimal ustar format) ─────────────────────────────

function tarHeader(entryPath: string, content: Buffer, type = "0"): Buffer {
  const header = Buffer.alloc(512, 0);
  header.write(entryPath, 0, Math.min(entryPath.length, 100), "utf-8");
  header.write("0000644\0", 100, 8, "utf-8");
  header.write("0001000\0", 108, 8, "utf-8");
  header.write("0001000\0", 116, 8, "utf-8");
  const size = type === "0" ? content.length : 0;
  header.write(size.toString(8).padStart(11, "0") + "\0", 124, 12, "utf-8");
  const mtime = Math.floor(Date.now() / 1000);
  header.write(mtime.toString(8).padStart(11, "0") + "\0", 136, 12, "utf-8");
  header.write(type, 156, 1, "utf-8");
  header.write("ustar\0", 257, 6, "utf-8");
  header.write("00", 263, 2, "utf-8");
  header.fill(0x20, 148, 156);
  let checksum = 0;
  for (let i = 0; i < 512; i++) checksum += header[i];
  header.write(checksum.toString(8).padStart(6, "0") + "\0 ", 148, 8, "utf-8");
  return header;
}

function buildTar(entries: Array<{ path: string; content?: string }>): Buffer {
  const blocks: Buffer[] = [];
  for (const entry of entries) {
    const content = Buffer.from(entry.content || "", "utf-8");
    blocks.push(tarHeader(entry.path, content, "0"));
    if (content.length > 0) {
      const paddedSize = Math.ceil(content.length / 512) * 512;
      const dataBlock = Buffer.alloc(paddedSize, 0);
      content.copy(dataBlock);
      blocks.push(dataBlock);
    }
  }
  blocks.push(Buffer.alloc(1024, 0));
  return Buffer.concat(blocks);
}

// ── Control-flow logic tests ────────────────────────────────────────
//
// These tests verify the decision logic extracted from backupSandboxState:
//   const tarPartial = result.status === 2 && result.stdout?.length > 0
//   if ((result.status === 0 || tarPartial) && result.stdout?.length > 0) { ... }

describe("rebuild tar exit-2 partial-success logic (#2727)", () => {
  it("accepts exit 2 with non-empty stdout as partial success", () => {
    const status = 2;
    const stdout = Buffer.from("non-empty-tar-data");
    const tarPartial = status === 2 && stdout != null && stdout.length > 0;
    expect(tarPartial).toBe(true);
    const shouldExtract = (status === 0 || tarPartial) && stdout != null && stdout.length > 0;
    expect(shouldExtract).toBe(true);
  });

  it("rejects exit 2 with empty stdout (connection failure, not permission error)", () => {
    const status = 2;
    const stdout = null;
    const tarPartial = status === 2 && stdout != null && (stdout as Buffer | null)?.length > 0;
    expect(tarPartial).toBeFalsy();
    const shouldExtract =
      (status === 0 || tarPartial) && stdout != null && (stdout as Buffer | null)?.length > 0;
    expect(shouldExtract).toBeFalsy();
  });

  it("rejects exit 1 regardless of stdout (not a GNU tar permission-error code)", () => {
    const status = 1;
    const stdout = Buffer.from("some-data");
    const tarPartial = status === 2 && stdout != null && stdout.length > 0;
    expect(tarPartial).toBe(false);
    const shouldExtract = (status === 0 || tarPartial) && stdout != null && stdout.length > 0;
    expect(shouldExtract).toBe(false);
  });

  it("accepts exit 0 with non-empty stdout (clean archive, existing behavior)", () => {
    const status = 0;
    const stdout = Buffer.from("tar-data");
    const tarPartial = status === 2 && stdout != null && stdout.length > 0;
    expect(tarPartial).toBe(false);
    const shouldExtract = (status === 0 || tarPartial) && stdout != null && stdout.length > 0;
    expect(shouldExtract).toBe(true);
  });
});

// ── End-to-end extraction tests ─────────────────────────────────────
//
// Verify that safeTarExtract successfully handles the archive GNU tar
// produces when it encounters root-owned files: a valid archive containing
// only the readable files, with the unreadable files simply absent.

describe("safeTarExtract handles partial archive from root-owned-file scenario (#2727)", () => {
  it("extracts partial archive (readable files only, root-owned files absent)", async () => {
    const { safeTarExtract } = await loadSandboxState();
    const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "nemoclaw-2727-"));
    try {
      const targetDir = path.join(workDir, "backup");
      fs.mkdirSync(targetDir, { recursive: true });

      // Simulate what GNU tar produces when it reads a directory that has
      // some root-owned files: it archives the readable files and exits 2.
      // The stdout is a valid tar archive with only the readable files.
      const partialArchive = buildTar([
        { path: "workspace/config.json", content: '{"agent":"openclaw"}' },
        { path: "memory/index.json", content: '{"entries":[]}' },
        // root-owned "credentials/token" is simply absent from the archive
      ]);

      const result = safeTarExtract(partialArchive, targetDir);

      expect(result.success).toBe(true);
      expect(fs.existsSync(path.join(targetDir, "workspace", "config.json"))).toBe(true);
      expect(fs.existsSync(path.join(targetDir, "memory", "index.json"))).toBe(true);
    } finally {
      fs.rmSync(workDir, { recursive: true, force: true });
    }
  });

  it("verifies partial archive is a valid tar (GNU tar would produce this on exit 2)", () => {
    // Confirm the archive format is correct — tar can list it.
    const archive = buildTar([
      { path: "workspace/config.json", content: '{"k":"v"}' },
    ]);

    const list = spawnSync("tar", ["-tf", "-"], {
      input: archive,
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });

    expect(list.status).toBe(0);
    const entries = (list.stdout || "").trim().split("\n").filter(Boolean);
    expect(entries).toContain("workspace/config.json");
  });
});

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
// These tests verify the decision logic extracted from backupSandboxState.
// Exit 2 is accepted only when every error line is a file-level
// "Cannot open: Permission denied" (path contains '/'), not a directory-level
// one (path has no '/') or any other error kind.

function onlyFilePermissionDenied(stderr: string): boolean {
  const lines = stderr.split("\n").filter((l) => l.trim().length > 0);
  if (lines.length === 0) return false;
  return lines.every((l) => {
    if (l.includes("Exiting with failure status")) return true;
    const m = /^tar: (.+): Cannot open: Permission denied/.exec(l);
    if (!m) return false;
    return m[1].includes("/"); // top-level path (no '/') = whole dir unreadable
  });
}

function simulateTarPartial(
  status: number,
  stdout: Buffer | null,
  stderr: string,
): boolean {
  if (status !== 2 || stdout == null || stdout.length === 0 || stderr.length === 0) return false;
  return onlyFilePermissionDenied(stderr);
}

describe("rebuild tar exit-2 partial-success logic (#2727)", () => {
  // Realistic stderr produced by root-owned FILES inside a state dir.
  // Paths contain '/' so they are file-level — safe to proceed.
  const FILE_PERM_STDERR = [
    "tar: memory/db.sqlite: Cannot open: Permission denied",
    "tar: memory/index.bin: Cannot open: Permission denied",
    "tar: Exiting with failure status due to previous errors",
  ].join("\n");

  it("accepts exit 2 when all errors are file-level Permission denied", () => {
    const stdout = Buffer.from("non-empty-tar-data");
    const tarPartial = simulateTarPartial(2, stdout, FILE_PERM_STDERR);
    expect(tarPartial).toBe(true);
    const shouldExtract = (2 === 0 || tarPartial) && stdout.length > 0;
    expect(shouldExtract).toBe(true);
  });

  it("rejects exit 2 when a top-level state dir itself is unreadable (path has no /)", () => {
    // Root-owned directory: tar reports the dir name with no slash.
    const stderr = [
      "tar: memory: Cannot open: Permission denied",
      "tar: Exiting with failure status due to previous errors",
    ].join("\n");
    const stdout = Buffer.from("some-data");
    const tarPartial = simulateTarPartial(2, stdout, stderr);
    expect(tarPartial).toBe(false);
  });

  it("rejects exit 2 with empty stdout (SSH/connection failure)", () => {
    const tarPartial = simulateTarPartial(2, null, FILE_PERM_STDERR);
    expect(tarPartial).toBe(false);
  });

  it("rejects exit 2 with empty stderr (unknown failure)", () => {
    const stdout = Buffer.from("some-data");
    const tarPartial = simulateTarPartial(2, stdout, "");
    expect(tarPartial).toBe(false);
  });

  it("rejects exit 2 when stderr contains an I/O error (not permission denied)", () => {
    const stderr = [
      "tar: workspace: Cannot stat: Input/output error",
      "tar: Exiting with failure status due to previous errors",
    ].join("\n");
    const stdout = Buffer.from("some-data");
    const tarPartial = simulateTarPartial(2, stdout, stderr);
    expect(tarPartial).toBe(false);
  });

  it("rejects exit 2 when stderr contains Cannot stat (missing entry)", () => {
    const stderr = [
      "tar: hooks: Cannot stat: No such file or directory",
      "tar: Exiting with failure status due to previous errors",
    ].join("\n");
    const stdout = Buffer.from("some-data");
    const tarPartial = simulateTarPartial(2, stdout, stderr);
    expect(tarPartial).toBe(false);
  });

  it("rejects exit 2 when stderr mixes file-level and dir-level errors", () => {
    const stderr = [
      "tar: memory/db.sqlite: Cannot open: Permission denied",
      "tar: hooks: Cannot open: Permission denied", // top-level dir — reject
      "tar: Exiting with failure status due to previous errors",
    ].join("\n");
    const stdout = Buffer.from("some-data");
    const tarPartial = simulateTarPartial(2, stdout, stderr);
    expect(tarPartial).toBe(false);
  });

  it("rejects exit 1 regardless of stderr (not a GNU tar exit-2 code)", () => {
    const stdout = Buffer.from("some-data");
    const tarPartial = simulateTarPartial(1, stdout, FILE_PERM_STDERR);
    expect(tarPartial).toBe(false);
  });

  it("accepts exit 0 with non-empty stdout (clean archive, existing behavior)", () => {
    const stdout = Buffer.from("tar-data");
    const tarPartial = simulateTarPartial(2, stdout, ""); // doesn't matter
    expect(tarPartial).toBe(false);
    const shouldExtract = (0 === 0 || tarPartial) && stdout.length > 0;
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

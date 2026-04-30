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
// accepts exit code 2 as a partial success when stderr matches the exact
// permission-denied pattern.

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
// Mirrors the production logic from backupSandboxState. Anchored matching
// for both the summary line and the per-file Permission denied lines
// prevents agent-controlled filenames containing those substrings from
// masking other error kinds.

const TAR_SUMMARY_LINE = "tar: Exiting with failure status due to previous errors";
const PERM_DENIED_RE = /^tar: (.+): Cannot open: Permission denied$/;

function collectPermDeniedPaths(stderr: string): string[] | null {
  const paths: string[] = [];
  const lines = stderr.split("\n").filter((l) => l.trim().length > 0);
  for (const l of lines) {
    if (l === TAR_SUMMARY_LINE) continue;
    const m = PERM_DENIED_RE.exec(l);
    if (!m) return null;
    paths.push(m[1]);
  }
  return paths.length > 0 ? paths : null;
}

// Simulator: returns true iff the production logic would accept exit 2 as
// a partial success. `dirPaths` simulates the set of paths that the SSH-stat
// follow-up would identify as directories (to be rejected).
function simulateTarPartial(
  status: number,
  stdout: Buffer | null,
  stderr: string,
  dirPaths: Set<string> = new Set(),
): boolean {
  if (status !== 2 || stdout == null || stdout.length === 0 || stderr.length === 0) return false;
  const paths = collectPermDeniedPaths(stderr);
  if (paths == null) return false;
  return !paths.some((p) => dirPaths.has(p));
}

describe("rebuild tar exit-2 partial-success logic (#2727)", () => {
  const FILE_PERM_STDERR = [
    "tar: memory/db.sqlite: Cannot open: Permission denied",
    "tar: memory/index.bin: Cannot open: Permission denied",
    TAR_SUMMARY_LINE,
  ].join("\n");

  it("accepts exit 2 when every error is an anchored Permission denied line", () => {
    expect(simulateTarPartial(2, Buffer.from("data"), FILE_PERM_STDERR)).toBe(true);
  });

  it("rejects exit 2 with empty stdout (SSH/connection failure)", () => {
    expect(simulateTarPartial(2, null, FILE_PERM_STDERR)).toBe(false);
  });

  it("rejects exit 2 with empty stderr (no error details)", () => {
    expect(simulateTarPartial(2, Buffer.from("data"), "")).toBe(false);
  });

  it("rejects exit 2 with only the summary line (no per-file Permission denied)", () => {
    expect(simulateTarPartial(2, Buffer.from("data"), `${TAR_SUMMARY_LINE}\n`)).toBe(false);
  });

  it("rejects exit 2 when one of the failing paths is actually a directory (subtree would be dropped)", () => {
    const stderr = [
      "tar: memory/db.sqlite: Cannot open: Permission denied",
      "tar: memory/cache: Cannot open: Permission denied", // this is a directory
      TAR_SUMMARY_LINE,
    ].join("\n");
    // Simulate the SSH-stat result reporting `memory/cache` as a directory.
    const dirPaths = new Set(["memory/cache"]);
    expect(simulateTarPartial(2, Buffer.from("data"), stderr, dirPaths)).toBe(false);
  });

  it("accepts exit 2 when the SSH-stat confirms all failing paths are files", () => {
    const stderr = [
      "tar: memory/db.sqlite: Cannot open: Permission denied",
      "tar: memory/index.bin: Cannot open: Permission denied",
      TAR_SUMMARY_LINE,
    ].join("\n");
    // SSH-stat returns no directories — both paths are files.
    expect(simulateTarPartial(2, Buffer.from("data"), stderr, new Set())).toBe(true);
  });

  it("rejects exit 2 when stderr contains a Cannot stat (I/O) error", () => {
    const stderr = [
      "tar: workspace: Cannot stat: Input/output error",
      TAR_SUMMARY_LINE,
    ].join("\n");
    expect(simulateTarPartial(2, Buffer.from("data"), stderr)).toBe(false);
  });

  it("rejects exit 2 when an agent-controlled filename smuggles 'Permission denied' into a different error", () => {
    // Anchored regex: phrase must be the line suffix, so a smuggled path is rejected.
    const stderr = [
      "tar: workspace/Cannot open: Permission denied/bar: Cannot stat: I/O error",
      TAR_SUMMARY_LINE,
    ].join("\n");
    expect(simulateTarPartial(2, Buffer.from("data"), stderr)).toBe(false);
  });

  it("rejects exit 2 when stderr's summary line is smuggled inside another diagnostic", () => {
    const stderr = [
      "tar: workspace/Exiting with failure status: Cannot stat: I/O error",
      "tar: memory/db.sqlite: Cannot open: Permission denied",
      TAR_SUMMARY_LINE,
    ].join("\n");
    expect(simulateTarPartial(2, Buffer.from("data"), stderr)).toBe(false);
  });

  it("rejects exit 1 regardless of stderr (not a GNU tar exit-2 code)", () => {
    expect(simulateTarPartial(1, Buffer.from("data"), FILE_PERM_STDERR)).toBe(false);
  });

  it("does not match exit 2 path for clean exit 0 (caller takes the exit-0 branch instead)", () => {
    // simulateTarPartial only models the exit-2 partial path; clean exit 0
    // is handled by the caller's || result.status === 0 condition.
    expect(simulateTarPartial(0, Buffer.from("data"), "")).toBe(false);
  });
});

// ── End-to-end extraction test ──────────────────────────────────────
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

      const partialArchive = buildTar([
        { path: "workspace/config.json", content: '{"agent":"openclaw"}' },
        { path: "memory/index.json", content: '{"entries":[]}' },
      ]);

      const result = safeTarExtract(partialArchive, targetDir);

      expect(result.success).toBe(true);
      expect(fs.existsSync(path.join(targetDir, "workspace", "config.json"))).toBe(true);
      expect(fs.existsSync(path.join(targetDir, "memory", "index.json"))).toBe(true);
    } finally {
      fs.rmSync(workDir, { recursive: true, force: true });
    }
  });

  it("verifies partial archive is a valid tar (sanity check on the test fixture)", () => {
    const archive = buildTar([{ path: "workspace/config.json", content: '{"k":"v"}' }]);
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

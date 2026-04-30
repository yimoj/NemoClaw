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
// Mirrors the production logic from backupSandboxState.
//
// The SSH command embeds a directory enumeration in stderr (between
// DIRS_BEGIN/END sentinels) before running tar. After tar completes we
// list the archive's contents and verify every enumerated directory is
// present. A missing directory → silently dropped subtree → abort.

const DIRS_BEGIN = "NEMOCLAW_DIRS_BEGIN_2727";
const DIRS_END = "NEMOCLAW_DIRS_END_2727";

function parseEnumeratedDirs(stderr: string, dir: string): { ok: boolean; relDirs: string[] } {
  const beginIdx = stderr.indexOf(DIRS_BEGIN);
  const endIdx = stderr.indexOf(DIRS_END);
  if (beginIdx < 0 || endIdx <= beginIdx) return { ok: false, relDirs: [] };
  const slice = stderr.substring(beginIdx + DIRS_BEGIN.length, endIdx);
  const abs = slice.split("\n").filter((d) => d.trim().length > 0);
  const prefix = `${dir.replace(/\/+$/, "")}/`;
  const rel = abs
    .map((d) => (d.startsWith(prefix) ? d.slice(prefix.length) : null))
    .filter((d): d is string => d != null && d.length > 0);
  return { ok: true, relDirs: rel };
}

function onlyPermDeniedErrors(stderr: string): boolean {
  // Strip enumeration block.
  const beginIdx = stderr.indexOf(DIRS_BEGIN);
  const endIdx = stderr.indexOf(DIRS_END);
  let scan = stderr;
  if (beginIdx >= 0 && endIdx > beginIdx) {
    scan = stderr.substring(0, beginIdx) + stderr.substring(endIdx + DIRS_END.length);
  }
  const lines = scan.split("\n").filter((l) => l.trim().length > 0);
  let sawPermDenied = false;
  for (const l of lines) {
    if (l.includes("Exiting with failure status")) continue;
    if (l.includes(DIRS_BEGIN) || l.includes(DIRS_END)) continue;
    if (!l.includes("Cannot open: Permission denied")) return false;
    sawPermDenied = true;
  }
  return sawPermDenied;
}

interface SimulateInput {
  status: number;
  stdout: Buffer | null;
  stderr: string;
  archivedDirs: Set<string>; // tar -t output for stdout, directories only
  dir: string;               // base dir, e.g. "/sandbox/.openclaw-data"
}

function simulate(input: SimulateInput): { partial: boolean; clean: boolean } {
  const { status, stdout, stderr, archivedDirs, dir } = input;
  const enumeration = parseEnumeratedDirs(stderr, dir);
  const missingDirs = enumeration.relDirs.filter((d) => !archivedDirs.has(d));
  // Empty enumeration = enumeration lost (e.g. shell redirection bug);
  // since existingDirs is non-empty, find should always emit at least those.
  const allDirsArchived =
    enumeration.ok && enumeration.relDirs.length > 0 && missingDirs.length === 0;
  const partial =
    status === 2 &&
    allDirsArchived &&
    stdout != null &&
    stdout.length > 0 &&
    onlyPermDeniedErrors(stderr);
  const clean =
    status === 0 &&
    allDirsArchived &&
    stdout != null &&
    stdout.length > 0;
  return { partial, clean };
}

describe("rebuild tar exit-2 partial-success logic (#2727)", () => {
  const DIR = "/sandbox/.openclaw-data";

  // Build a stderr that includes a successful directory enumeration block.
  function withEnum(absDirs: string[], extraLines: string[] = []): string {
    return [
      DIRS_BEGIN,
      ...absDirs,
      DIRS_END,
      ...extraLines,
    ].join("\n");
  }

  const FILE_PERM_LINES = [
    "tar: memory/db.sqlite: Cannot open: Permission denied",
    "tar: memory/index.bin: Cannot open: Permission denied",
    "tar: Exiting with failure status due to previous errors",
  ];

  it("accepts exit 2 when all enumerated dirs are in archive and only Permission denied errors appear", () => {
    const stderr = withEnum([`${DIR}/workspace`, `${DIR}/memory`], FILE_PERM_LINES);
    const { partial } = simulate({
      status: 2,
      stdout: Buffer.from("data"),
      stderr,
      archivedDirs: new Set(["workspace", "memory"]),
      dir: DIR,
    });
    expect(partial).toBe(true);
  });

  it("rejects exit 2 when an enumerated directory is missing from the archive (subtree dropped)", () => {
    const stderr = withEnum([`${DIR}/workspace`, `${DIR}/memory`, `${DIR}/memory/cache`], FILE_PERM_LINES);
    const { partial } = simulate({
      status: 2,
      stdout: Buffer.from("data"),
      stderr,
      archivedDirs: new Set(["workspace", "memory"]), // memory/cache absent
      dir: DIR,
    });
    expect(partial).toBe(false);
  });

  it("rejects exit 2 when the top-level state dir itself is missing from the archive", () => {
    const stderr = withEnum([`${DIR}/workspace`, `${DIR}/memory`], FILE_PERM_LINES);
    const { partial } = simulate({
      status: 2,
      stdout: Buffer.from("data"),
      stderr,
      archivedDirs: new Set(["workspace"]), // memory missing entirely
      dir: DIR,
    });
    expect(partial).toBe(false);
  });

  it("rejects exit 2 when stderr does not contain the enumeration sentinels (probe failed)", () => {
    const stderrNoEnum = FILE_PERM_LINES.join("\n");
    const { partial } = simulate({
      status: 2,
      stdout: Buffer.from("data"),
      stderr: stderrNoEnum,
      archivedDirs: new Set(["workspace", "memory"]),
      dir: DIR,
    });
    expect(partial).toBe(false);
  });

  it("rejects exit 2 when enumeration block is present but empty (e.g. shell redirection bug)", () => {
    // Sentinels present but no directories between them — would happen if
    // find's stdout was lost to /dev/null due to wrong redirection order.
    const stderr = withEnum([], FILE_PERM_LINES);
    const { partial } = simulate({
      status: 2,
      stdout: Buffer.from("data"),
      stderr,
      archivedDirs: new Set(["workspace", "memory"]),
      dir: DIR,
    });
    expect(partial).toBe(false);
  });

  it("rejects exit 2 when stderr contains only the summary line (no Permission denied lines)", () => {
    const stderr = withEnum([`${DIR}/workspace`, `${DIR}/memory`], [
      "tar: Exiting with failure status due to previous errors",
    ]);
    const { partial } = simulate({
      status: 2,
      stdout: Buffer.from("data"),
      stderr,
      archivedDirs: new Set(["workspace", "memory"]),
      dir: DIR,
    });
    expect(partial).toBe(false);
  });

  it("rejects exit 2 when stderr contains a non-permission-denied error (I/O error)", () => {
    const stderr = withEnum([`${DIR}/workspace`, `${DIR}/memory`], [
      "tar: workspace: Cannot stat: Input/output error",
      "tar: Exiting with failure status due to previous errors",
    ]);
    const { partial } = simulate({
      status: 2,
      stdout: Buffer.from("data"),
      stderr,
      archivedDirs: new Set(["workspace", "memory"]),
      dir: DIR,
    });
    expect(partial).toBe(false);
  });

  it("rejects exit 2 with empty stdout (SSH/tar failure)", () => {
    const stderr = withEnum([`${DIR}/workspace`], FILE_PERM_LINES);
    const { partial } = simulate({
      status: 2,
      stdout: null,
      stderr,
      archivedDirs: new Set(),
      dir: DIR,
    });
    expect(partial).toBe(false);
  });

  it("rejects exit 1 regardless of other inputs (not a GNU tar exit-2 code)", () => {
    const stderr = withEnum([`${DIR}/workspace`], FILE_PERM_LINES);
    const { partial } = simulate({
      status: 1,
      stdout: Buffer.from("data"),
      stderr,
      archivedDirs: new Set(["workspace"]),
      dir: DIR,
    });
    expect(partial).toBe(false);
  });

  it("accepts exit 0 (clean archive) when all enumerated dirs are present", () => {
    const stderr = withEnum([`${DIR}/workspace`, `${DIR}/memory`]);
    const { clean } = simulate({
      status: 0,
      stdout: Buffer.from("clean-archive"),
      stderr,
      archivedDirs: new Set(["workspace", "memory"]),
      dir: DIR,
    });
    expect(clean).toBe(true);
  });

  it("rejects exit 0 when enumeration claims a dir not in the archive (deeper problem)", () => {
    const stderr = withEnum([`${DIR}/workspace`, `${DIR}/memory`]);
    const { clean } = simulate({
      status: 0,
      stdout: Buffer.from("clean-archive"),
      stderr,
      archivedDirs: new Set(["workspace"]),
      dir: DIR,
    });
    expect(clean).toBe(false);
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

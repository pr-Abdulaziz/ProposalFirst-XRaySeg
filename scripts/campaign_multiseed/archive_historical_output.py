#!/usr/bin/env python
"""Archive the historical generated results under output/, verify, then remove them.

Scope is an explicit list; nothing is discovered by globbing. Symlinks are never
followed: a symlink inside the scope aborts the run. The archive is verified
member by member (size and SHA-256 against the pre-archive inventory) before any
active copy is removed, and the kept part of output/ is hashed before and after
to prove it was not touched.

    python scripts/campaign_multiseed/archive_historical_output.py --dry-run
    python scripts/campaign_multiseed/archive_historical_output.py --execute
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path("/nfs/ahalqahtani/ProposalFirst-XRaySeg").resolve()
OUTPUT = ROOT / "output"
STAMP = "2026-09-14"
ARCHIVE_DIR = ROOT / "archive" / f"historical_output_{STAMP}"
TARBALL = ARCHIVE_DIR / f"historical_output_{STAMP}.tar.gz"

# Generated results from the previous campaigns (reports, metrics, predictions,
# audit records, manifests, job logs). Paths are relative to the project root.
SCOPE = [
    "output/audit",
    "output/dev_campaign",
    "output/error_analysis",
    "output/full_split_campaign",
    "output/logs",
    "output/reports",
    "output/pixray_source_before_training.sha256",
    "output/pdf/pixray_evaluation_report.tex",
    "output/pdf/pixray_evaluation_report.html",
    "output/pdf/pixray_evaluation_report_11arms.tex.bak",
]
# Source material, handoffs and architecture documentation stay in place.
KEEP = [
    "output/docs",
    "output/diagrams",
    "output/imagegen",
    "output/pdf/Occlusion-aware_highlighted.pdf",
    "output/pdf/PIDray_paper_highlighted.pdf",
    "output/pdf/cluster_usage_guide.pdf",
    "output/pdf/cluster_usage_guide.tex",
    "output/pdf/pidray_occ_baselines_focus_plan.pdf",
    "output/pdf/pidray_occ_baselines_focus_plan.tex",
    "output/pdf/pixray_experiment_architecture_guide.pdf",
    "output/pdf/pixray_experiment_architecture_guide.tex",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_inside(rel: str) -> Path:
    path = ROOT / rel
    if path.is_symlink():
        raise SystemExit(f"ABORT: {rel} is a symlink; refusing to follow it")
    real = Path(os.path.realpath(path))
    if not str(real).startswith(str(OUTPUT) + os.sep):
        raise SystemExit(f"ABORT: {rel} resolves to {real}, outside {OUTPUT}")
    if not path.exists():
        raise SystemExit(f"ABORT: {rel} does not exist")
    return real


def inventory(rels: list[str]) -> dict[str, dict]:
    files: dict[str, dict] = {}
    for rel in rels:
        top = resolve_inside(rel)
        walk = [top] if top.is_file() else sorted(p for p in top.rglob("*"))
        for p in walk:
            if p.is_symlink():
                raise SystemExit(f"ABORT: symlink inside scope: {p}")
            if p.is_dir():
                continue
            st = p.stat()
            files[str(p.relative_to(ROOT))] = dict(
                resolved=str(p), size=st.st_size, sha256=sha256(p),
                mtime=dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"))
    return files


def references(scoped: list[str]) -> dict[str, list[str]]:
    """Code/docs outside output/ that name an archived path (recorded, not edited)."""
    needles = [s for s in scoped]
    hits: dict[str, list[str]] = {}
    for base in ("scripts", "tools", "projects", "tests", "docs", "configs", "README.md"):
        base_path = ROOT / base
        candidates = [base_path] if base_path.is_file() else base_path.rglob("*")
        for p in candidates:
            if p.is_symlink() or not p.is_file() or "__pycache__" in p.parts:
                continue
            if p.suffix not in {".py", ".sh", ".md", ".tex", ".json", ".txt", ""}:
                continue
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            found = sorted({n for n in needles if n in text})
            if found:
                hits[str(p.relative_to(ROOT))] = found
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    if TARBALL.exists():
        raise SystemExit(f"ABORT: {TARBALL} already exists; refusing to overwrite")
    archived = inventory(SCOPE)
    kept_before = inventory(KEEP)
    top_level = sorted(p.name for p in OUTPUT.iterdir())
    covered = {Path(r).parts[1] for r in SCOPE + KEEP}
    unclassified = [n for n in top_level if n not in covered]
    if unclassified:
        raise SystemExit(f"ABORT: unclassified entries under output/: {unclassified}")
    pdf_unclassified = sorted(
        str(p.relative_to(ROOT)) for p in (OUTPUT / "pdf").iterdir()
        if str(p.relative_to(ROOT)) not in SCOPE + KEEP)
    if pdf_unclassified:
        raise SystemExit(f"ABORT: unclassified files under output/pdf: {pdf_unclassified}")

    total = sum(v["size"] for v in archived.values())
    print(f"scope: {len(SCOPE)} entries, {len(archived)} files, {total/1e6:.1f} MB")
    print(f"kept:  {len(KEEP)} entries, {len(kept_before)} files")
    if args.dry_run:
        return 0

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=False)
    with tarfile.open(TARBALL, "w:gz", format=tarfile.PAX_FORMAT, dereference=False) as tar:
        for rel in sorted(archived):
            tar.add(ROOT / rel, arcname=rel, recursive=False)

    # Verify every member against the pre-archive inventory before removing anything.
    seen = {}
    with tarfile.open(TARBALL, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                raise SystemExit(f"ABORT: unexpected non-file member {member.name}")
            h = hashlib.sha256()
            f = tar.extractfile(member)
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
            seen[member.name] = dict(size=member.size, sha256=h.hexdigest())
    problems = [r for r in archived if r not in seen
                or seen[r]["size"] != archived[r]["size"]
                or seen[r]["sha256"] != archived[r]["sha256"]]
    extra = [r for r in seen if r not in archived]
    if problems or extra:
        raise SystemExit(f"ABORT: archive verification failed: {problems[:5]} extra={extra[:5]}")
    print(f"archive verified: {len(seen)} members match size and sha256")

    manifest = dict(
        created=dt.datetime.now().isoformat(timespec="seconds"),
        purpose=("Historical generated results removed from the active output/ tree "
                 "before the PIXray multi-seed campaign. Paths inside the tarball are "
                 "project-relative; restore with: tar -xzf <tarball> -C <project root>."),
        project_root=str(ROOT),
        tarball=str(TARBALL.relative_to(ROOT)),
        tarball_sha256=sha256(TARBALL),
        tarball_size=TARBALL.stat().st_size,
        scope=SCOPE,
        kept_in_output=KEEP,
        files=archived,
        file_count=len(archived),
        total_bytes=total,
        code_references_to_archived_paths=references(SCOPE),
        not_in_scope=("work_dirs/ (historical checkpoints and predictions), data/, "
                      "pretrained/, checkpoints/, splits/, source, configs and scripts "
                      "are outside this cleanup and were not touched."),
    )
    (ARCHIVE_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))

    for rel in SCOPE:
        real = resolve_inside(rel)
        if real.is_dir():
            shutil.rmtree(real)
        else:
            real.unlink()

    gone = [r for r in SCOPE if (ROOT / r).exists()]
    kept_after = inventory(KEEP)
    changed = [r for r in kept_before if kept_before[r]["sha256"] != kept_after.get(r, {}).get("sha256")]
    verification = dict(
        removed_paths_absent=not gone, still_present=gone,
        kept_files_before=len(kept_before), kept_files_after=len(kept_after),
        kept_files_changed=changed,
        output_top_level_after=sorted(p.name for p in OUTPUT.iterdir()),
        output_pdf_after=sorted(p.name for p in (OUTPUT / "pdf").iterdir()),
    )
    (ARCHIVE_DIR / "CLEANUP_VERIFICATION.json").write_text(json.dumps(verification, indent=1))
    print(json.dumps(verification, indent=1))
    return 0 if not gone and not changed and len(kept_before) == len(kept_after) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
fetch_data.py -- download the pinned third-party inputs and build ROOT/data.

  1. Shallow-fetches the two source repositories at their pinned commits into ROOT/ext/:
       ROOT/ext/datuoniao_cot-dashboard   github.com/datuoniao/cot-dashboard   @ d9e10fda...
           data/positions.json, data/prices.json  -> tools/build_data.py
       ROOT/ext/accomplish999_crude-oil   github.com/accomplish999/crude-oil   @ 379e29d8...
           data/series/CFTC_CL.json               -> lenses/crude2026/basis_check_ext.py
  2. Downloads the EIA Cushing WTI spot series (datasets/oil-prices, data/wti-daily.csv) to
     ROOT/data/wti-daily.csv. The default URL is pinned to the datasets/oil-prices commit whose
     file is byte-identical to the copy used for the published results (downloaded 2026-10-16
     from the main branch, last row 2026-10-14).
  3. Runs tools/build_data.py, which writes ROOT/data/cot_weekly.csv and ROOT/data/prices_daily.csv.
  4. Compares SHA-256 hashes with the files used for the published results (warning only).

Everything downloaded is treated as untrusted data: it is only parsed (json / csv), never
executed or imported. No submodules, no LFS smudge, no hooks; build_data.py runs with
python3 -I so nothing from ROOT/ext or the current directory can be imported.

Usage:
    python3 -I tools/fetch_data.py [--root ROOT] [--wti-url URL] [--skip-build]
ROOT defaults to $LARRY_COT_ROOT, else the package directory (parent of tools/).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

REPOS = [
    # (dest dir under ROOT/ext, clone URL, pinned commit)
    ("datuoniao_cot-dashboard", "https://github.com/datuoniao/cot-dashboard",
     "d9e10fdacab5d5328b8d75619e23b2b620610014"),
    ("accomplish999_crude-oil", "https://github.com/accomplish999/crude-oil",
     "379e29d8a78c05222813decaa2fc840c35672b27"),
]
WTI_COMMIT = "c24526249395d1df291d2a6a63a510939cf65556"   # datasets/oil-prices, 2026-10-16
WTI_URL = f"https://raw.githubusercontent.com/datasets/oil-prices/{WTI_COMMIT}/data/wti-daily.csv"
WTI_URL_LATEST = "https://raw.githubusercontent.com/datasets/oil-prices/main/data/wti-daily.csv"

# SHA-256 of the exact files behind the published results (relative to ROOT).
EXPECTED_SHA256 = {
    "ext/datuoniao_cot-dashboard/data/positions.json":
        "c77acd4a2735e36c9e65e425671749080864df14b1f6e004c189469b2b76c190",
    "ext/datuoniao_cot-dashboard/data/prices.json":
        "08c3be973beb0035e47dc3ba3d7e54e5d5d1d02ef2c69dd16938f18d2b150165",
    "ext/accomplish999_crude-oil/data/series/CFTC_CL.json":
        "f2d5d2df9643acc1d537046e1fa4982ef7c2eed21e5ed80e9d126386a05b9168",
    "data/wti-daily.csv":
        "08e3029c618fdcba5da53b64b9911e49788da658d9ff361210575ab5064f854e",
    "data/cot_weekly.csv":
        "1a5006ad643bb077de087ce541ca242617f19d895f9b377458179f060f4fff14",
    "data/prices_daily.csv":
        "e93305bb45f6d95e11c43b4a3658cea15893bc570ce1de226745fa3b0bfe030d",
}

# Environment for every git call: no LFS downloads, no interactive prompts, ignore any
# system/global hooks path. Nothing in the fetched tree can configure or run code.
GIT_ENV = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
GIT_SAFE = ["-c", "core.hooksPath=/dev/null", "-c", "submodule.recurse=false",
            "-c", "advice.detachedHead=false"]


def git(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    cmd = ["git", *GIT_SAFE, *args]
    r = subprocess.run(cmd, cwd=cwd, env=GIT_ENV, check=True, text=True,
                       stdout=subprocess.PIPE if capture else None)
    return (r.stdout or "").strip()


def head_of(dest: Path) -> str | None:
    if not (dest / ".git").is_dir():
        return None
    try:
        return git("-C", str(dest), "rev-parse", "HEAD", capture=True)
    except subprocess.CalledProcessError:
        return None


def fetch_repo(ext: Path, name: str, url: str, commit: str) -> None:
    dest = ext / name
    have = head_of(dest)
    if have == commit:
        print(f"[ext] {name}: already at {commit[:12]}, skipping")
        return
    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"[ext] {dest} exists but is not a checkout of {commit} (HEAD={have}). "
                 f"Delete it and rerun.")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[ext] {name}: shallow fetch {url} @ {commit}")
    git("init", "-q", str(dest))
    git("-C", str(dest), "remote", "add", "origin", url)
    git("-C", str(dest), "fetch", "-q", "--depth", "1", "--no-tags",
        "--no-recurse-submodules", "origin", commit)
    git("-C", str(dest), "checkout", "-q", "--detach", "FETCH_HEAD")
    got = head_of(dest)
    if got != commit:
        sys.exit(f"[ext] {name}: HEAD is {got}, expected {commit}")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:   # honours HTTPS_PROXY
        body = resp.read()
    # Light sanity check that this is the expected plain CSV ("Date,Price" header).
    head = body[:64].decode("utf-8", errors="replace").splitlines()[0].strip()
    if head != "Date,Price":
        sys.exit(f"[data] unexpected header in {url}: {head!r}")
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp.write(body)
    os.chmod(tmp.name, 0o644)
    os.replace(tmp.name, dest)
    print(f"[data] wrote {dest} ({len(body)} bytes)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hashes(root: Path) -> int:
    bad = 0
    for rel, want in EXPECTED_SHA256.items():
        p = root / rel
        if not p.exists():
            print(f"[hash] MISSING   {rel}")
            bad += 1
            continue
        ok = sha256(p) == want
        bad += not ok
        print(f"[hash] {'OK      ' if ok else 'MISMATCH'}  {rel}")
    if bad:
        print("[hash] WARNING: inputs differ from the ones behind the published results; "
              "numbers may differ.")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="fetch pinned inputs and build ROOT/data")
    ap.add_argument("--root", default=os.environ.get("LARRY_COT_ROOT") or str(PKG),
                    help="root dir (default: $LARRY_COT_ROOT or the package dir)")
    ap.add_argument("--wti-url", default=WTI_URL,
                    help=f"wti-daily.csv URL (default: pinned commit). Latest: {WTI_URL_LATEST}")
    ap.add_argument("--skip-build", action="store_true", help="do not run build_data.py")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)   # keep our lines in order with git/build output

    root = Path(args.root).resolve()
    ext = root / "ext"
    ext.mkdir(parents=True, exist_ok=True)
    for name, url, commit in REPOS:
        fetch_repo(ext, name, url, commit)
    download(args.wti_url, root / "data" / "wti-daily.csv")

    if not args.skip_build:
        print("[build] tools/build_data.py")
        subprocess.run([sys.executable, "-I", str(PKG / "tools" / "build_data.py"), str(root)],
                       check=True, cwd=str(root))
    check_hashes(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

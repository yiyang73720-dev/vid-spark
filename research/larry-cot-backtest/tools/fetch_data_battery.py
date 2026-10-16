#!/usr/bin/env python3
"""
fetch_data_battery.py -- download the two extra third-party inputs of the fundamentals battery
(battery/, spec v2) into ROOT/ext2/, at pinned commits, with a sparse checkout of only the files
the battery reads.

  ROOT/ext2/robcarver17_pysystemtrade    github.com/robcarver17/pysystemtrade      @ bbe29e19...
      data/futures/multiple_prices_csv/{CRUDE_W,GOLD,SILVER,SOYBEAN,SUGAR11,CORN,WHEAT,COPPER,
      GAS_US,HEATOIL,GASOILINE}.csv   -> battery/scout/scout_check.py (curve_daily.csv) and
                                         battery/tool_seasonality/seas_data.py (roll indices)
      data/futures/csvconfig/rollconfig.csv, LICENSE  (not read by code: the roll offsets cited
                                         in the seasonality caveats, and the GPL-3 licence)
  ROOT/ext2/bgithub1_cme_open_interest   github.com/bgithub1/cme_open_interest     @ a5772c97...
      cot_history.csv, cot_new_history.csv  (CFTC legacy / disaggregated COT incl. non-reportable
                                         positions) -> battery/scout/scout_check.py
                                         (cot_smalltrader_weekly.csv)

The pysystemtrade repository is large, so each repository is fetched with depth 1 and
--filter=blob:none, and only the listed paths are checked out (core.sparseCheckout): git
downloads just those blobs. The battery also reads the v1 inputs (ROOT/data, ROOT/ext); those
come from tools/fetch_data.py. Their SHA-256 for the files the battery reads is checked here too.

Everything downloaded is treated as untrusted data: it is only parsed (csv / json), never
executed or imported. No submodules, no LFS smudge, no hooks (same git settings as
tools/fetch_data.py).

Usage:
    python3 -I tools/fetch_data_battery.py [--root ROOT] [--check-only]
ROOT defaults to $LARRY_COT_ROOT, else the package directory (parent of tools/).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "tools"))
import fetch_data as v1   # noqa: E402  same git safety settings, git() / head_of() / sha256()

PST_DATA = "data/futures/multiple_prices_csv"
PST_INSTRUMENTS = ("CRUDE_W", "GOLD", "SILVER", "SOYBEAN", "SUGAR11", "CORN", "WHEAT", "COPPER",
                   "GAS_US", "HEATOIL", "GASOILINE")

REPOS2 = [
    # (dest dir under ROOT/ext2, clone URL, pinned commit, paths to check out)
    ("robcarver17_pysystemtrade", "https://github.com/robcarver17/pysystemtrade",
     "bbe29e19dab73271570c94b4ee7e2f182d4b9ef1",          # 2026-10-08
     [f"{PST_DATA}/{i}.csv" for i in PST_INSTRUMENTS]
     + ["data/futures/csvconfig/rollconfig.csv", "LICENSE"]),
    ("bgithub1_cme_open_interest", "https://github.com/bgithub1/cme_open_interest",
     "a5772c97fcb58f06ed8c945fcfec1b7e6ddc5263",          # 2021-07-13
     ["cot_history.csv", "cot_new_history.csv"]),
]

# SHA-256 of the exact files behind the published battery results (relative to ROOT).
EXPECTED_SHA256 = {
    f"ext2/robcarver17_pysystemtrade/{PST_DATA}/CRUDE_W.csv":
        "64a8c9f46998830eb242695d9d519f2547d20820d222b2c127ebd71f213958d1",
    f"ext2/robcarver17_pysystemtrade/{PST_DATA}/GOLD.csv":
        "5e02e932c5c17aa7e5807e942866a6a0f6f867c1bdca6698acf3609d05af6bdd",
    f"ext2/robcarver17_pysystemtrade/{PST_DATA}/SILVER.csv":
        "28cfb68f999378d35e53291debf0b818b3912cf70056063547e582f9c3731b04",
    f"ext2/robcarver17_pysystemtrade/{PST_DATA}/SOYBEAN.csv":
        "911a09de70eca193a756bf88a12eca98eb195a248f3007ade5573c029f0b1bf4",
    f"ext2/robcarver17_pysystemtrade/{PST_DATA}/SUGAR11.csv":
        "a6d626bd104d15b782a0cc9a0b5ecd7467e3068cae0e9a36970e6abdab429258",
    f"ext2/robcarver17_pysystemtrade/{PST_DATA}/CORN.csv":
        "47183abccaa9e260e2c934acab3819ee0d4edcaf3fa7fc0c4a96abd933452e1c",
    f"ext2/robcarver17_pysystemtrade/{PST_DATA}/WHEAT.csv":
        "a038800ee43cc86d30c0cf2989225f5190ebd9a2499dd10bca83528703ab912e",
    f"ext2/robcarver17_pysystemtrade/{PST_DATA}/COPPER.csv":
        "8c31c285cb11f85d08097398c3babe5a87ec555c0cbebf7b1cce9456ed71a2fb",
    f"ext2/robcarver17_pysystemtrade/{PST_DATA}/GAS_US.csv":
        "8d1b2ad71db68931ace326244f3527ea893a63876e4c3eb21709ed696f1da7c6",
    f"ext2/robcarver17_pysystemtrade/{PST_DATA}/HEATOIL.csv":
        "7ec8e0d1a69927e58a990f9f05f4e1f07ea53dd6a28b70a5bc7b844888e414da",
    f"ext2/robcarver17_pysystemtrade/{PST_DATA}/GASOILINE.csv":
        "f318010f57b22d38e482b3ca03f0232c7b2a4b0a7a188086fe6e0ebc8b04397a",
    "ext2/robcarver17_pysystemtrade/data/futures/csvconfig/rollconfig.csv":
        "8c752d19cd6ce9f512caa888189a9264793a98f68d99791fe05b4d7f93485422",
    "ext2/bgithub1_cme_open_interest/cot_history.csv":
        "84b93b9b38500bbb7303a1ab31538dea7696a0b00862c41303205f2332107b6d",
    "ext2/bgithub1_cme_open_interest/cot_new_history.csv":
        "830e701820c4919ce5668369d44e3e6a522ac57c0d6bcac8e440bcee987bdf60",
}

# v1 inputs (tools/fetch_data.py) that the battery also reads, beyond the ones fetch_data.py
# checks itself: EIA / FRED series of the accomplish999 archive at its pinned commit.
V1_SERIES = "ext/accomplish999_crude-oil/data/series"
EXPECTED_SHA256_V1_EXTRA = {f"{V1_SERIES}/{sid}.json": h for sid, h in {
    "RCLC1": "0b26c694836a43c9f0910686a791f2dcfe4cd2f577aa8137ad632b2e486694cc",   # NYMEX WTI c1..c4
    "RCLC2": "7996451cb94f38a4c82ab4d8eca519f7be2ca3ab898c8fc403d770aa1957e568",
    "RCLC3": "7840c4adf052ebfcdc52de7ba10491eb00146c8f9f240ce306849c857d7f331d",
    "RCLC4": "7158a2da2785ae53b27700e26ed208caceb6b58d80651fe9c93ea83851d277cf",
    "HO_F1": "496dd3219000b0fad1ebd0baa0dee7579d3da1e0d316855bb6aa187b81824ea5",   # heating oil c1
    "RBOB_F1": "15f5f7a0881ee0d8380bc41eb2c2fb672cc50ff237e0db384ed49eb935cd76ce",  # RBOB c1
    "HO_SPOT": "d208d89c48c0bec49605438626b64ce9b816ad8dc041b8ef88bf5b7ed0699243",
    "GAS_SPOT": "dff1b37108ea35bceeeddd0dc88721afbaad1a8833f32ab204ebd1c6457cfa23",
    "RBRTE": "a8014e84c0cb473a918d369d277441e9741ce150c1db65ebf94cc0a405a35729",    # Brent spot
    "DGS10": "1f70d5b124f9e630928f6f472361b51b15370b381a3e9ad3f73bf3bec5986907",    # 10y yield
    "DTWEXBGS": "04043a435b38cbea70b07daa84f9e445c23014d056912f6fb84558cadeba8db3",  # broad dollar
    "WCESTUS1": "74910b92d777de06d46b18e856d1b409b327fffdf982b3a642b3f8537f839cdc",  # crude stocks
    "CFTC_CL": "f2d5d2df9643acc1d537046e1fa4982ef7c2eed21e5ed80e9d126386a05b9168",   # COT, fut.-only
}.items()}


def fetch_sparse(ext2: Path, name: str, url: str, commit: str, paths: list[str]) -> None:
    """Shallow, blob-filtered fetch of `commit` with a sparse checkout of `paths` only."""
    dest = ext2 / name
    have = v1.head_of(dest)
    if have == commit and all((dest / p).is_file() for p in paths):
        print(f"[ext2] {name}: already at {commit[:12]}, skipping")
        return
    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"[ext2] {dest} exists but is not a complete checkout of {commit} "
                 f"(HEAD={have}). Delete it and rerun.")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[ext2] {name}: shallow sparse fetch {url} @ {commit} ({len(paths)} files)")
    v1.git("init", "-q", str(dest))
    v1.git("-C", str(dest), "remote", "add", "origin", url)
    v1.git("-C", str(dest), "config", "core.sparseCheckout", "true")
    info = dest / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    # Non-cone patterns, anchored at the repository root: exactly these files.
    (info / "sparse-checkout").write_text("".join(f"/{p}\n" for p in paths), encoding="utf-8")
    v1.git("-C", str(dest), "fetch", "-q", "--depth", "1", "--filter=blob:none", "--no-tags",
           "--no-recurse-submodules", "origin", commit)
    v1.git("-C", str(dest), "checkout", "-q", "--detach", "FETCH_HEAD")
    got = v1.head_of(dest)
    if got != commit:
        sys.exit(f"[ext2] {name}: HEAD is {got}, expected {commit}")
    missing = [p for p in paths if not (dest / p).is_file()]
    if missing:
        sys.exit(f"[ext2] {name}: not checked out: {missing}")


def check_hashes(root: Path, expected: dict[str, str], tag: str) -> int:
    bad = 0
    for rel, want in expected.items():
        p = root / rel
        if not p.exists():
            print(f"[{tag}] MISSING   {rel}")
            bad += 1
            continue
        ok = v1.sha256(p) == want
        bad += not ok
        print(f"[{tag}] {'OK      ' if ok else 'MISMATCH'}  {rel}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="fetch the battery's pinned inputs into ROOT/ext2")
    ap.add_argument("--root", default=os.environ.get("LARRY_COT_ROOT") or str(PKG),
                    help="root dir (default: $LARRY_COT_ROOT or the package dir)")
    ap.add_argument("--check-only", action="store_true",
                    help="do not download; only compare SHA-256 hashes")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)   # keep our lines in order with git output

    root = Path(args.root).resolve()
    if not args.check_only:
        ext2 = root / "ext2"
        ext2.mkdir(parents=True, exist_ok=True)
        for name, url, commit, paths in REPOS2:
            fetch_sparse(ext2, name, url, commit, paths)
    bad = check_hashes(root, EXPECTED_SHA256, "hash ext2")
    bad += check_hashes(root, EXPECTED_SHA256_V1_EXTRA, "hash v1  ")
    if bad:
        print("[hash] WARNING: inputs differ from the ones behind the published battery results; "
              "numbers may differ. (v1 inputs come from tools/fetch_data.py.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# Rerun the whole Larry Williams COT backtest end to end.
#
#   ./run_all.sh [--skip-fetch] [--crosscheck]
#
#   --skip-fetch   do not download anything; ROOT/data (and ROOT/ext for basis_check_ext.py)
#                  must already exist
#   --crosscheck   also run the two independent implementations (crosscheck/implA, implB) and
#                  engine/verify_engine.py, which diffs them against the engine
#
# ROOT (data/, ext/, work/) = $LARRY_COT_ROOT, else this directory. Outputs go to ROOT/work/.
# Every Python step runs with -I (isolated mode; -B: no .pyc): third-party files under ROOT/ext and
# ROOT/data are only parsed, never imported or executed.
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LARRY_COT_ROOT:-$PKG}"
mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"
export LARRY_COT_ROOT="$ROOT"
PY="${PYTHON:-python3}"

SKIP_FETCH=0
CROSSCHECK=0
for arg in "$@"; do
  case "$arg" in
    --skip-fetch) SKIP_FETCH=1 ;;
    --crosscheck) CROSSCHECK=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

LOGS="$ROOT/work/logs"
mkdir -p "$LOGS"
step() {  # step <log name> <script> [args...]
  local name="$1"; shift
  echo "=== $name: $*"
  "$PY" -I -B "$@" 2>&1 | tee "$LOGS/$name.log"
}

echo "package: $PKG"
echo "root:    $ROOT"

# 0. data
if [ "$SKIP_FETCH" -eq 0 ]; then
  step 00_fetch_data "$PKG/tools/fetch_data.py" --root "$ROOT"
fi
for f in cot_weekly.csv prices_daily.csv wti-daily.csv; do
  [ -f "$ROOT/data/$f" ] || { echo "missing $ROOT/data/$f (run without --skip-fetch)" >&2; exit 1; }
done

# 1. engine (canonical.csv, crude_timeseries.csv, pooled.csv, canonical_exact_p.csv)
step 01_engine "$PKG/engine/larry_cot.py" --root "$ROOT"

# 1b. optional independent cross-check
if [ "$CROSSCHECK" -eq 1 ]; then
  step 01a_implA "$PKG/crosscheck/implA/backtest_canonical.py" "$ROOT"
  step 01b_implB "$PKG/crosscheck/implB/backtest_b.py" "$ROOT"
  step 01c_verify_engine "$PKG/engine/verify_engine.py" "$ROOT"
fi

# 2. lenses, in the order they were originally run
step 02_strategy_run     "$PKG/lenses/strategy/run_strategies.py"
step 03_strategy_check   "$PKG/lenses/strategy/check_strategies.py"
step 04_robustness       "$PKG/lenses/robustness/run_all.py"
step 05_crude2026_basis  "$PKG/lenses/crude2026/basis_check_ext.py"   # reads ROOT/ext (CFTC_CL.json)
step 06_comparators      "$PKG/lenses/comparators/run_comparators.py"
step 07_crude2026_case   "$PKG/lenses/crude2026/crude2026_case.py"

# 3. compact JSON for the report page -> ROOT/work/report/data.json
step 08_report_data "$PKG/tools/make_report_data.py" "$ROOT/work" "$ROOT/work/report/data.json"

echo "done. outputs in $ROOT/work, logs in $LOGS"

#!/usr/bin/env bash
# Rerun the Larry Williams fundamentals battery (spec v2, SPEC2.md) end to end.
#
#   battery/run_battery.sh [--skip-fetch]
#
#   --skip-fetch   do not download anything; ROOT/data, ROOT/ext (v1 inputs) and ROOT/ext2
#                  (battery inputs) must already exist
#
# ROOT = $LARRY_COT_ROOT, else the package directory (the parent of battery/), the same root as
# ../run_all.sh. Inputs: ROOT/data + ROOT/ext (tools/fetch_data.py), ROOT/ext2
# (tools/fetch_data_battery.py). The battery also reads four v1 outputs (ROOT/work/engine/*.csv,
# ROOT/work/lens_strategy/metrics.csv); if they are missing, the v1 engine and strategy lens are
# run first. Battery outputs go to ROOT/work2/<step>/, logs to ROOT/work2/logs/.
# Every Python step runs with -I (isolated mode; -B: no .pyc): third-party files under ROOT/ext,
# ROOT/ext2 and ROOT/data are only parsed, never imported or executed.
set -euo pipefail

BAT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$BAT/.." && pwd)"
ROOT="${LARRY_COT_ROOT:-$PKG}"
mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"
export LARRY_COT_ROOT="$ROOT"
PY="${PYTHON:-python3}"

SKIP_FETCH=0
for arg in "$@"; do
  case "$arg" in
    --skip-fetch) SKIP_FETCH=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

W2="$ROOT/work2"
LOGS="$W2/logs"
mkdir -p "$LOGS"
step() {  # step <log name> <script> [args...]
  local name="$1"; shift
  echo "=== $name: $*"
  "$PY" -I -B "$@" 2>&1 | tee "$LOGS/$name.log"
}

echo "package: $PKG"
echo "battery: $BAT"
echo "root:    $ROOT"

# 0. inputs ------------------------------------------------------------------------------------
V1_IN=(data/cot_weekly.csv data/prices_daily.csv data/wti-daily.csv
       ext/accomplish999_crude-oil/data/series/CFTC_CL.json)
v1_inputs_ok() { local f; for f in "${V1_IN[@]}"; do [ -f "$ROOT/$f" ] || return 1; done; }
if [ "$SKIP_FETCH" -eq 0 ]; then
  if v1_inputs_ok; then
    echo "=== 00a_fetch_v1: ROOT/data and ROOT/ext present, skipping (tools/fetch_data.py)"
  else
    step 00a_fetch_v1 "$PKG/tools/fetch_data.py" --root "$ROOT"
  fi
  step 00b_fetch_battery "$PKG/tools/fetch_data_battery.py" --root "$ROOT"
fi
v1_inputs_ok || { echo "missing v1 inputs under $ROOT (run without --skip-fetch)" >&2; exit 1; }
for f in robcarver17_pysystemtrade/data/futures/multiple_prices_csv/CRUDE_W.csv \
         bgithub1_cme_open_interest/cot_history.csv; do
  [ -f "$ROOT/ext2/$f" ] || { echo "missing $ROOT/ext2/$f (run without --skip-fetch)" >&2; exit 1; }
done

# v1 outputs read by the battery (harness self-test, rank, family BH audit)
V1_OUT=(work/engine/canonical.csv work/engine/canonical_exact_p.csv work/engine/pooled.csv
        work/lens_strategy/metrics.csv)
v1_missing=0
for f in "${V1_OUT[@]}"; do [ -f "$ROOT/$f" ] || v1_missing=1; done
if [ "$v1_missing" -eq 1 ]; then
  step 00c_v1_engine   "$PKG/engine/larry_cot.py" --root "$ROOT"
  step 00d_v1_strategy "$PKG/lenses/strategy/run_strategies.py"
fi

# 1. shared harness: reproduces the v1 COT study bit for bit -> ROOT/work2/harness
step 01_harness_selftest "$BAT/harness/selftest_harness.py"

# 2. data scout: tidy curve_daily.csv / cot_smalltrader_weekly.csv -> ROOT/work2/scout
step 02_scout "$BAT/scout/scout_check.py"

# 3. the five tools, each: pre-registered run -> independent checks (order as originally run;
#    a step that writes into results.json comes after the run that creates it)
step 03a_premium_run          "$BAT/tool_premium/run_premium.py"
step 03b_premium_verify       "$BAT/tool_premium/verify_primary.py"
step 03c_premium_independent  "$BAT/tool_premium/verifier/verify_independent.py"
step 03d_premium_checks       "$BAT/tool_premium/verifier/verifier_checks.py"     # adds "verification"

step 04a_seasonality_run      "$BAT/tool_seasonality/run_seasonality.py"
step 04b_seasonality_posthoc  "$BAT/tool_seasonality/posthoc_gasoline.py"
step 04c_seasonality_vprimary "$BAT/tool_seasonality/verify/verify_primary.py"
step 04d_seasonality_vtrade   "$BAT/tool_seasonality/verify/verify_tradable.py"
step 04e_seasonality_vroll    "$BAT/tool_seasonality/verify/verify_roll_proxy.py"
step 04f_seasonality_vs1      "$BAT/tool_seasonality/verify/verify_s1.py"
step 04g_seasonality_apply    "$BAT/tool_seasonality/verify/apply_verification.py"  # adds "verification"

step 05a_valuation_smoke      "$BAT/tool_valuation/smoke_inputs.py"
step 05b_valuation_run        "$BAT/tool_valuation/run_valuation.py"
step 05c_valuation_verify     "$BAT/tool_valuation/verify_primary.py"
step 05d_valuation_audit      "$BAT/tool_valuation/audit_valuation.py"

# run_open_interest.py embeds posthoc_gold.json and adv_verify_oi_harnessltd.json, so those first
step 06a_oi_posthoc_gold      "$BAT/tool_open_interest/posthoc_gold.py"
step 06b_oi_adv_verify        "$BAT/tool_open_interest/adv_verify_oi.py"
step 06c_oi_adv_verify_ltd    "$BAT/tool_open_interest/adv_verify_oi.py" --harness-ltd
step 06d_oi_run               "$BAT/tool_open_interest/run_open_interest.py"
step 06e_oi_verify            "$BAT/tool_open_interest/verify_primary.py"
step 06f_oi_adv_robust        "$BAT/tool_open_interest/adv_verify_robust.py"
step 06g_oi_adv_wf            "$BAT/tool_open_interest/adv_verify_wf.py"
step 06h_oi_adv_compare_ltd   "$BAT/tool_open_interest/adv_compare_ltd.py"

step 07a_intermarket_run      "$BAT/tool_intermarket/run_intermarket.py"
step 07b_intermarket_verify   "$BAT/tool_intermarket/verify_primary.py"
step 07c_intermarket_adv      "$BAT/tool_intermarket/adv_verify_intermarket.py"  # adds a block

# 4. rank: every primary rebuilt exactly -> confluence -> ranking + family-wide BH
step 08_rank_verify_rebuild   "$BAT/rank/verify_rebuild.py"
step 09_confluence            "$BAT/rank/run_confluence.py"
step 10_rank                  "$BAT/rank/run_rank.py"

# 5. final adversarial audit (reads the rank tables), then the rank step once more so that
#    ROOT/work2/rank/results.json carries the audit lines, as published
step 11a_audit_premium        "$BAT/audit_final/audit_premium_independent.py"
step 11b_audit_premium_trades "$BAT/audit_final/audit_premium_strategy_trades.py"
step 11c_audit_family_bh      "$BAT/audit_final/audit_family_bh.py"
step 11d_audit_confluence     "$BAT/audit_final/audit_confluence.py"
step 12_rank_final            "$BAT/rank/run_rank.py"

echo "done. headline: $W2/rank/results.json and $W2/rank/primary_table.csv; logs in $LOGS"

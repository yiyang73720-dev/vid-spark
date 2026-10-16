#!/usr/bin/env python3
"""
apply_verification -- write the adversarial verification into results.json (in place).

Reads the outputs of the independent checks (ROOT/work2/tool_seasonality/verify/)
    verify_primary.json      (primary re-derivation, robustness, split halves)
    verify_tradable.json     (claims C4-C6, gasoline basis, index alignment)
    verify_roll_proxy.json   (front-month proxies, sugar roll-day artifact)
and the per-market S1 re-simulation from verify_s1.py, then
  * adds a "verification" block to results.json (all numbers copied from those files),
  * brings results.json "caveats" in line with run_seasonality.py (corrected off-calendar caveat,
    seas_data.PSYS_CAVEATS), so the file matches what a fresh run would write.
No statistic of the original run is changed: the re-derivation reproduced them.

Run after the three verify_*.py scripts (and again after any re-run of run_seasonality.py,
which rewrites results.json):  python3 -I apply_verification.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import seas_data as sd    # noqa: E402  (PSYS_CAVEATS)
import verify_primary as vp  # noqa: E402
import verify_s1          # noqa: E402

fh = vp.fh
RESULTS = fh.WORK2 / "tool_seasonality" / "results.json"     # outputs of run_seasonality.py
OLD_OFFCAL = ("Standard circular-shift p is conservative for a seasonal tool: shifts by ~52 weeks "
              "keep the calendar aligned. perm_p_offcal excludes those offsets (diagnostic, "
              "pre-registered).")
NEW_OFFCAL = ("Shifts of about 52 weeks keep a seasonal signal aligned with the calendar, so the "
              "standard circular-shift p can be conservative; perm_p_offcal excludes those offsets "
              "(diagnostic, pre-registered). It is not uniformly smaller (gasoline: larger), and it "
              "also credits ANY calendar-locked pattern -- roll-day jumps of unadjusted futures, "
              "spot grade switches -- so a small perm_p_offcal is not evidence of a tradable "
              "seasonal.")


def load(name: str) -> dict:
    return json.loads((vp.HERE / name).read_text())      # ROOT/work2/tool_seasonality/verify


def main():
    res = json.loads(RESULTS.read_text())
    vpj, vtj, vrj = load("verify_primary.json"), load("verify_tradable.json"), load("verify_roll_proxy.json")
    s1 = [verify_s1.s1(n, L) for _, n, L in vp.MARKETS]

    res["verification"] = dict(
        note=("Adversarial verification. Independent re-implementation in verify/ (no code shared "
              "with seasonal.py / run_seasonality.py; fund_harness used only to load series). All "
              "numbers below are copied from verify/*.json; perm p values there use their own "
              "generator (seed 2024)."),
        prereg_check=("PREREG.md (08:44:43) and its sha256 (08:44:48) predate seasonal.py (08:48), "
                      "run_seasonality.py (08:54) and results.json (08:59); the sha256 matches the "
                      "file. The headline is the pre-registered rule, not the best grid cell "
                      "(several grid cells have larger excess)."),
        primary_rederived=vpj["primary_pooled_net_h8"],
        primary_rederived_matches_results=("n, mean, excess, hit, T and period equal results.json "
                                           "exactly; episodes differ only by convention (see "
                                           "episode_convention)."),
        episode_convention=("results.json counts 793 episodes with the harness rule (a week whose "
                            "signal is undefined ends an episode); counting runs over eligible weeks "
                            f"only gives {vpj['n_episodes_eligible_only']}."),
        min_days_decision_minus_latest_analog_exit=vpj["min_days_decision_minus_last_analog_exit"],
        per_market_rederived=vpj["per_market"],
        common_offset_null=("Shifting all markets by the SAME offset (keeps cross-market correlation) "
                            "gives perm p = " + str(vpj["primary_pooled_net_h8"]["perm_p_common_offset_seed2024"])),
        robustness=dict(
            share_of_total_excess_from_gasoline_spot=vpj["share_of_total_excess_from_gasoline"],
            excess_symmetric_trim_top_and_bottom_N_episodes=vpj["excess_symmetric_trim_episodes"],
            excess_drop_single_year=vpj["drop_years"],
            excess_drop_2008_2014_2020=vpj["drop_2008_2014_2020"],
            split_halves_2013_10_21_with_and_without_gasoline=vpj["split_halves_2013_10_21"],
            without_gasoline_sugar_soybean=vpj["without_gasoline_sugar_soybean"],
        ),
        artifacts=dict(
            sugar_v1_mean_abs_daily_return=vrj["sugar_v1_mean_abs_ret"],
            primary_with_sugar_roll_days_zeroed=vrj["primary_sugar_rolls_zeroed"],
            primary_with_sugar_roll_days_zeroed_without_gasoline=vrj["primary_sugar_rolls_zeroed_without_gasoline"],
            gas_spot_minus_futures_log_drift_per_year_by_quarter=vtj["gas_spot_minus_fut_log_drift_per_year_by_quarter"],
            gas_spot_vs_rbob_f1_basis_step_week8_to_week10=vtj["gas_basis_step_week8_to_week10"],
            gas_spot_vs_rbob_f1_basis_step_week35_to_week36=vtj["gas_basis_step_week35_to_week36"],
        ),
        tradability_rederived=dict(C4_full=vtj["C4_tradable_full"], C4_2001_2024=vtj["C4_tradable_2001_2024"],
                                   C4_per_market=vtj["C4_per_market"],
                                   C5=vtj["C5_primary_signal_on_tradable"], C6=vtj["C6_gasoline_setup_weeks"]),
        psys_index_alignment_vs_eia_contract1=vtj["index_vs_eia_contract1_from_2006"],
        tradability_front_month_proxies=dict(
            heatoil_front_L15=vrj["heatoil_front_L15"], heatoil_front_L10=vrj["heatoil_front_L10"],
            gasoline_rbob_front_L10=vrj["gasoline_rbob_front_L10"],
            sugar_front_L15=vrj["sugar_front_L15"], sugar_front_L10=vrj["sugar_front_L10"],
            pooled_psys_as_author=vrj["pooled_tradable_psys_as_author"],
            pooled_ho_front=vrj["pooled_tradable_ho_front"],
            pooled_ho_front_sugar_front=vrj["pooled_tradable_ho_front_sugar_front"]),
        s1_long_short_rederived=s1,
        verdict=dict(
            preregistered_label_reproduced="positive EV, fragile",
            verifier_practical_verdict="no edge",
            reason=("The pre-registered ladder is applied correctly (c1 and c2 true, c3 false, c4 "
                    "true). But the significant primary is measured on price series nobody can "
                    "trade as such: over half its excess comes from NY Harbor gasoline spot, whose "
                    "Dec-Mar/Sep-Oct pattern is a spot-vs-futures basis and grade cycle, and part "
                    "of the sugar edge is roll-day jumps of the unadjusted SB series. With sugar "
                    "roll days removed and gasoline spot dropped the primary is not significant; on "
                    "every tradable construction (author's indices, front-month proxies) the pooled "
                    "rule has no edge. Reported as: no demonstrated tradable edge."),
        ),
    )
    caveats = [NEW_OFFCAL if c == OLD_OFFCAL else c for c in res["caveats"]]
    caveats += [c for c in sd.PSYS_CAVEATS if c not in caveats]
    res["caveats"] = caveats
    RESULTS.write_text(json.dumps(fh.to_jsonable(res), indent=1))
    print("updated", RESULTS)


if __name__ == "__main__":
    main()

"""
Phase 3: run walk-forward calibration, then refit a final config on all
available seasons and re-generate the Elo rating history with it.

Usage:
    python -m cfb_elo.run_backtest
"""
from __future__ import annotations

import dataclasses
import json
import logging

import pandas as pd

from cfb_elo import config
from cfb_elo.backtest import brier_score, calibration_table, grid_search_walk_forward, home_rows
from cfb_elo.elo import EloConfig, EloRatingSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# k_factor, regression_pct, and home_field_advantage converged to stable
# interior values across two widened passes (~25-35, ~0.2-0.5, 45).
#
# mov_cap never converged -- it kept getting pushed to whatever ceiling the
# grid allowed (35 -> 49 -> 70), but a direct comparison with a 95% CI on
# Brier score showed the entire mov_cap=28..70 range is statistically
# indistinguishable (spread of 0.0004 vs. a CI half-width of 0.0059): the
# log-dampening in the MOV multiplier already suppresses blowout impact, so
# the explicit cap stops mattering once it's above ~30-35. It's fixed at 35
# (top of the original 28-35 target range) rather than grid-selected, since
# a higher value would be picking noise while defeating the point of having
# a cap at all.
GRID = dict(
    k_values=(20, 25, 30, 35),
    mov_cap_values=(35,),
    regression_values=(0.2, 0.3, 0.35, 0.4, 0.5),
    hfa_values=(35, 40, 45, 50, 55),
)


def main() -> None:
    games_path = config.DATA_PROCESSED_DIR / f"games_{config.START_SEASON}_{config.END_SEASON}.parquet"
    games = pd.read_parquet(games_path)

    logger.info("=== Walk-forward validation (train on prior seasons only) ===")
    folds = grid_search_walk_forward(games, **GRID)

    print("\nWalk-forward results (FBS-vs-FBS games only):")
    fold_df = pd.DataFrame([
        {
            "test_season": f.test_season,
            "k_factor": f.chosen_config.k_factor,
            "mov_cap": f.chosen_config.mov_cap,
            "regression_pct": f.chosen_config.regression_pct,
            "home_field_advantage": f.chosen_config.home_field_advantage,
            "train_brier": round(f.train_brier, 4),
            "test_brier": round(f.test_brier, 4),
            "baseline_brier_0.5": round(f.baseline_brier, 4),
        }
        for f in folds
    ])
    print(fold_df.to_string(index=False))

    avg_test_brier = fold_df["test_brier"].mean()
    avg_baseline = fold_df["baseline_brier_0.5"].mean()
    print(f"\nAverage out-of-sample Brier: {avg_test_brier:.4f}  (vs. always-guess-0.5 baseline: {avg_baseline:.4f})")

    # Locked from direct sensitivity sweeps (see conversation/commit history),
    # not a fresh grid-argmin: k_factor, home_field_advantage, and
    # regression_pct all showed genuine interior optima (U-shaped Brier vs.
    # parameter, degrading on both sides, well outside the ~+/-0.006 95% CI
    # noise band) at these values. mov_cap is flat above ~30 (log-dampening
    # already does the work) so it's kept at 35 -- the top of the originally
    # requested 28-35 range -- rather than an arbitrarily large grid-selected
    # value that would defeat the point of having a cap.
    final_cfg = EloConfig(k_factor=30, mov_cap=35, regression_pct=0.35, home_field_advantage=45)
    home_all = home_rows(EloRatingSystem(final_cfg).run(games))
    home_all = home_all[home_all["season"] > home_all["season"].min()]
    final_train_brier = brier_score(home_all["win_probability"], home_all["won"].astype(float))
    print(f"\nFinal locked config (fit on all {config.START_SEASON}-{config.END_SEASON} data):")
    print(final_cfg)
    print(f"In-sample Brier with this config: {final_train_brier:.4f}")

    cfg_path = config.DATA_PROCESSED_DIR / "elo_config.json"
    cfg_path.write_text(json.dumps(dataclasses.asdict(final_cfg), indent=2))
    logger.info("Wrote tuned config to %s", cfg_path)

    logger.info("Re-running Elo with tuned config over full history...")
    elo = EloRatingSystem(final_cfg)
    history = elo.run(games)
    history.to_parquet(config.DATA_PROCESSED_DIR / "elo_history.parquet", index=False)
    history.to_csv(config.DATA_PROCESSED_DIR / "elo_history.csv", index=False)
    logger.info("Wrote %d team-game rows using tuned config", len(history))

    home_fbs = history[history["is_home"] & (history["tier"] != "FCS") & (history["opponent_tier"] != "FCS")]
    table = calibration_table(home_fbs)
    print("\nCalibration (predicted win-prob bucket vs. actual win rate, FBS-vs-FBS games, all seasons):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

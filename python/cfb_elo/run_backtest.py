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
from cfb_elo.backtest import calibration_table, grid_search_walk_forward, select_final_config
from cfb_elo.elo import EloConfig, EloRatingSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Widened from the initial pass: mov_cap and home_field_advantage both hit
# the edge of the original grid (21-35 / 45-85) in every fold, meaning the
# true optimum was outside the search range rather than settled inside it.
GRID = dict(
    k_values=(10, 15, 20, 25, 30, 35),
    mov_cap_values=(21, 28, 35, 42, 49),
    regression_values=(0.1, 0.2, 0.35, 0.5, 0.65),
    hfa_values=(5, 25, 45, 65),
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

    logger.info("=== Final config: grid search over all seasons ===")
    final_cfg, final_train_brier = select_final_config(games, **GRID)
    print(f"\nFinal chosen config (fit on all {config.START_SEASON}-{config.END_SEASON} data):")
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

"""
Build a per-team win-count probability distribution (P(0 wins) ... P(12
wins)) from an already-generated Monte Carlo win-totals matrix. Kept
separate from run_phase4.py so it can be regenerated without re-running the
simulation.

Usage:
    python -m cfb_elo.win_distribution
"""
from __future__ import annotations

import logging

import pandas as pd

from cfb_elo import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_win_distribution(win_totals: pd.DataFrame, max_wins: int = 12) -> pd.DataFrame:
    """Returns a (n_teams x (max_wins+1)) DataFrame: rows are teams, columns
    are win counts 0..max_wins, values are P(that many wins) across sims."""
    n_sims = len(win_totals)
    dist = pd.DataFrame(
        {w: (win_totals == w).sum() / n_sims for w in range(max_wins + 1)},
        index=win_totals.columns,
    )
    dist.columns = [f"p_{w}_wins" for w in dist.columns]
    dist.index.name = "team"
    return dist


def main() -> None:
    win_totals = pd.read_parquet(config.DATA_PROCESSED_DIR / f"sim_{config.TARGET_SEASON}_win_totals.parquet")
    schedule = pd.read_parquet(config.DATA_PROCESSED_DIR / f"schedule_{config.TARGET_SEASON}.parquet")

    home = schedule[["home_team", "home_classification"]].rename(
        columns={"home_team": "team", "home_classification": "classification"})
    away = schedule[["away_team", "away_classification"]].rename(
        columns={"away_team": "team", "away_classification": "classification"})
    classification = pd.concat([home, away]).drop_duplicates("team").set_index("team")["classification"]

    fbs_teams = classification[classification == "fbs"].index
    max_wins = int(win_totals.max().max())

    dist = build_win_distribution(win_totals[fbs_teams], max_wins=max_wins)
    dist.insert(0, "mean_wins", win_totals[fbs_teams].mean().round(2))
    dist = dist.sort_values("mean_wins", ascending=False)

    out_path = config.DATA_PROCESSED_DIR / f"sim_{config.TARGET_SEASON}_win_distribution.csv"
    dist.round(4).to_csv(out_path)
    logger.info("Wrote win-count distribution for %d teams to %s", len(dist), out_path)

    print(f"\nWin-count distribution, top 10 teams by mean wins ({config.TARGET_SEASON}):")
    print(dist.head(10).to_string())

    # sanity check: probabilities for each team should sum to ~1
    row_sums = dist.drop(columns="mean_wins").sum(axis=1)
    off = (row_sums - 1.0).abs().max()
    logger.info("Max deviation from 1.0 across all teams' summed probabilities: %.6f", off)


if __name__ == "__main__":
    main()

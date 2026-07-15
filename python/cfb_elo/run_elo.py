"""
Run the Elo rating engine over the pulled game data and write out the
per-game rating history, plus a quick sanity-check printout.

Usage:
    python -m cfb_elo.run_elo
"""
from __future__ import annotations

import logging

import pandas as pd

from cfb_elo import config
from cfb_elo.elo import EloConfig, EloRatingSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    games_path = config.DATA_PROCESSED_DIR / f"games_{config.START_SEASON}_{config.END_SEASON}.parquet"
    games = pd.read_parquet(games_path)

    elo = EloRatingSystem(EloConfig())
    history = elo.run(games)

    stem = "elo_history"
    history.to_parquet(config.DATA_PROCESSED_DIR / f"{stem}.parquet", index=False)
    history.to_csv(config.DATA_PROCESSED_DIR / f"{stem}.csv", index=False)
    logger.info("Wrote %d team-game rows to %s.parquet / .csv", len(history), stem)

    latest_season = games["season"].max()
    final = (
        history[history["season"] == latest_season]
        .sort_values(["team", "start_date"])
        .groupby("team")
        .tail(1)
        .sort_values("post_game_rating", ascending=False)
    )
    print(f"\nTop 25 teams by end-of-{latest_season} Elo rating:")
    print(final[["team", "tier", "post_game_rating"]].head(25).to_string(index=False))

    print(f"\nBottom 10 teams by end-of-{latest_season} Elo rating:")
    print(final[["team", "tier", "post_game_rating"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()

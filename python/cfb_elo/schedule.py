"""
Phase 4: fetch an upcoming (not-yet-played) season's schedule from CFBD.

Unlike data_pull.py, incomplete games are kept here -- that's the entire
point. The schedule for a season that hasn't been played yet is nothing
BUT incomplete games (no scores), and the Monte Carlo simulation in
simulate.py needs exactly that: who plays whom, home/away/neutral, and
each team's conference for that season.

Regular season only by default -- posted win-total lines are almost always
based on regular-season games, excluding conference championship and
bowl/CFP games (which aren't determined until December anyway).
"""
from __future__ import annotations

import logging

import cfbd
import pandas as pd

from cfb_elo.data_pull import _ALIAS_TO_SNAKE, GAME_COLUMNS, _client

logger = logging.getLogger(__name__)


def fetch_schedule(season: int, season_type: cfbd.SeasonType = cfbd.SeasonType.REGULAR) -> pd.DataFrame:
    with _client() as api_client:
        games_api = cfbd.GamesApi(api_client)
        games = games_api.get_games(
            year=season,
            season_type=season_type,
            classification=cfbd.DivisionClassification.FBS,
        )

    if not games:
        raise RuntimeError(f"No schedule returned for {season} -- has it been published yet?")

    records = [g.to_dict() for g in games]
    df = pd.DataFrame.from_records(records).rename(columns=_ALIAS_TO_SNAKE)
    keep = [c for c in GAME_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True)

    n_completed = int(df["completed"].sum()) if "completed" in df.columns else 0
    logger.info("Fetched %d scheduled games for %s (%d already completed)", len(df), season, n_completed)
    return df

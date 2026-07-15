"""
Phase 1: pull historical FBS game data from CollegeFootballData.com.

One row per game (final score, home/away, date, season). Games where either
side is not FBS (i.e. an FCS opponent) are kept but flagged via
`is_fcs_game` rather than dropped, per the project spec.

Usage:
    python -m cfb_elo.data_pull
"""
from __future__ import annotations

import logging

import cfbd
import pandas as pd

from cfb_elo import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# snake_case columns we keep, after un-aliasing the API's camelCase response
GAME_COLUMNS = [
    "id", "season", "week", "season_type", "start_date", "completed",
    "neutral_site", "conference_game", "venue",
    "home_id", "home_team", "home_conference", "home_classification", "home_points",
    "away_id", "away_team", "away_conference", "away_classification", "away_points",
]

_ALIAS_TO_SNAKE = {
    "seasonType": "season_type", "startDate": "start_date",
    "neutralSite": "neutral_site", "conferenceGame": "conference_game",
    "homeId": "home_id", "homeTeam": "home_team", "homeConference": "home_conference",
    "homeClassification": "home_classification", "homePoints": "home_points",
    "awayId": "away_id", "awayTeam": "away_team", "awayConference": "away_conference",
    "awayClassification": "away_classification", "awayPoints": "away_points",
}


def _client() -> cfbd.ApiClient:
    if not config.CFBD_API_KEY:
        raise RuntimeError(
            "CFBD_API_KEY is not set. Copy python/.env.example to python/.env "
            "and add your key from https://collegefootballdata.com/key"
        )
    cfg = cfbd.Configuration(access_token=config.CFBD_API_KEY)
    return cfbd.ApiClient(cfg)


def fetch_season_games(games_api: cfbd.GamesApi, season: int) -> pd.DataFrame:
    """Pull all games (regular + postseason) involving an FBS team for one season."""
    games = games_api.get_games(
        year=season,
        season_type=cfbd.SeasonType.BOTH,
        classification=cfbd.DivisionClassification.FBS,
    )
    if not games:
        logger.warning("No games returned for season %s", season)
        return pd.DataFrame(columns=GAME_COLUMNS)

    records = [g.to_dict() for g in games]
    df = pd.DataFrame.from_records(records).rename(columns=_ALIAS_TO_SNAKE)
    keep = [c for c in GAME_COLUMNS if c in df.columns]
    return df[keep]


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True)
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")
    df["is_fcs_game"] = (df["home_classification"] != "fbs") | (df["away_classification"] != "fbs")
    df["margin"] = df["home_points"] - df["away_points"]
    df["home_win"] = df["margin"] > 0
    return df


def pull_all_seasons(seasons: list[int]) -> pd.DataFrame:
    frames = []
    with _client() as api_client:
        games_api = cfbd.GamesApi(api_client)
        for season in seasons:
            logger.info("Pulling season %s...", season)
            df = fetch_season_games(games_api, season)
            df.to_parquet(config.DATA_RAW_DIR / f"games_{season}.parquet", index=False)
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = add_derived_columns(combined)
    completed_mask = combined["completed"] == True  # noqa: E712
    dropped = (~completed_mask).sum()
    if dropped:
        logger.info("Dropping %d not-yet-completed games", dropped)
    return combined[completed_mask].reset_index(drop=True)


def main() -> None:
    combined = pull_all_seasons(config.SEASONS)
    stem = f"games_{config.START_SEASON}_{config.END_SEASON}"
    combined.to_parquet(config.DATA_PROCESSED_DIR / f"{stem}.parquet", index=False)
    combined.to_csv(config.DATA_PROCESSED_DIR / f"{stem}.csv", index=False)

    logger.info("Wrote %d games to %s.parquet / .csv", len(combined), stem)
    logger.info(
        "FCS-opponent games: %d / %d (%.1f%%)",
        combined["is_fcs_game"].sum(), len(combined),
        100 * combined["is_fcs_game"].mean(),
    )
    logger.info("Seasons: %s", sorted(combined["season"].unique().tolist()))


if __name__ == "__main__":
    main()

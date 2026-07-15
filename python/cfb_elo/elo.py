"""
Phase 2: Elo-style rating engine.

Ratings update game-by-game in chronological order. Between seasons, each
team's rating is regressed toward its conference tier's mean (P5 / G5 / FCS)
rather than reset. Tier is re-evaluated using the team's CURRENT season's
conference, so realignment is handled for free -- e.g. Oklahoma and Texas
moving Big 12 -> SEC in 2024, or the Pac-12's near-total dissolution, just
shows up as those teams' preseason ratings regressing toward a different
tier mean the next time they're seen.

All tunable knobs (K-factor, MOV cap, home-field advantage, regression %,
tier means) live on EloConfig. The defaults below are reasonable starting
points, not final values -- Phase 3 calibrates them by minimizing Brier
score walk-forward across seasons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

P5_CONFERENCES = {"ACC", "Big Ten", "Big 12", "SEC", "Pac-12"}
# "FBS Independents" mixes Notre Dame (P5-caliber schedule) with mid-major
# independents (UMass, UConn pre-Big 12, etc.) -- special-case by team below.
INDEPENDENT_P5_TEAMS = {"Notre Dame"}

TIER_P5, TIER_G5, TIER_FCS = "P5", "G5", "FCS"


@dataclass
class EloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 20.0
    mov_cap: float = 28.0                  # raw point margin is capped here before the MOV multiplier is computed
    home_field_advantage: float = 65.0     # Elo points added to the home team when computing win probability
    regression_pct: float = 0.35           # fraction of the way a team's rating is pulled toward its tier mean between seasons
    tier_means: dict = field(default_factory=lambda: {TIER_P5: 1600.0, TIER_G5: 1450.0, TIER_FCS: 1250.0})

    def tier_mean(self, tier: str) -> float:
        return self.tier_means.get(tier, self.initial_rating)


def classify_team_tier(team: str, conference: str | None, classification: str | None) -> str:
    """Map a team's (conference, classification) for one season to a rating tier."""
    if classification != "fbs":
        return TIER_FCS
    if conference in P5_CONFERENCES:
        return TIER_P5
    if conference == "FBS Independents" and team in INDEPENDENT_P5_TEAMS:
        return TIER_P5
    return TIER_G5


def mov_multiplier(margin: float, elo_diff: float, cap: float) -> float:
    """538-style margin-of-victory dampener: blowouts count for less, and the
    same blowout counts for more when scored by the underdog than the favorite.

    `margin` is the (unsigned) point margin, capped at `cap` before use.
    `elo_diff` is the pre-game rating gap between the game's winner and loser
    (i.e. how big an upset this was, if it was one).
    """
    capped = min(abs(margin), cap)
    return math.log(capped + 1) * (2.2 / (0.001 * elo_diff + 2.2))


@dataclass
class TeamGameResult:
    game_id: int
    season: int
    week: int
    start_date: object
    team: str
    opponent: str
    is_home: bool
    tier: str
    opponent_tier: str
    pre_game_rating: float
    post_game_rating: float
    win_probability: float
    won: bool
    points_for: float
    points_against: float


class EloRatingSystem:
    """Stateful rating tracker. Call `run()` once with the full chronological
    game history; `self.ratings` holds the latest rating per team afterward."""

    def __init__(self, config: EloConfig | None = None):
        self.config = config or EloConfig()
        self.ratings: dict[str, float] = {}
        self._last_season: dict[str, int] = {}

    def _preseason_adjust(self, team: str, season: int, tier: str) -> float:
        """Lazily regress a team's rating toward its tier mean the first time
        it's seen in a new season; initialize new teams at the tier mean."""
        tier_mean = self.config.tier_mean(tier)

        if team not in self.ratings:
            self.ratings[team] = tier_mean
            self._last_season[team] = season
            return self.ratings[team]

        if self._last_season[team] != season:
            prior = self.ratings[team]
            self.ratings[team] = tier_mean + (prior - tier_mean) * (1 - self.config.regression_pct)
            self._last_season[team] = season

        return self.ratings[team]

    def run(self, games: pd.DataFrame) -> pd.DataFrame:
        """Iterate games in chronological order, updating `self.ratings` in
        place. Returns a long-format DataFrame (one row per team per game)
        with pre/post ratings and win probability, for backtesting/analysis."""
        games = games.sort_values(["season", "week", "start_date", "id"]).reset_index(drop=True)

        rows: list[TeamGameResult] = []
        for g in games.itertuples(index=False):
            home_tier = classify_team_tier(g.home_team, g.home_conference, g.home_classification)
            away_tier = classify_team_tier(g.away_team, g.away_conference, g.away_classification)

            r_home = self._preseason_adjust(g.home_team, g.season, home_tier)
            r_away = self._preseason_adjust(g.away_team, g.season, away_tier)

            expected_home = 1.0 / (1.0 + 10 ** (-((r_home + self.config.home_field_advantage) - r_away) / 400.0))
            home_won = g.home_points > g.away_points
            actual_home = 1.0 if home_won else 0.0

            winner_rating, loser_rating = (r_home, r_away) if home_won else (r_away, r_home)
            multiplier = mov_multiplier(g.margin, winner_rating - loser_rating, self.config.mov_cap)

            delta = self.config.k_factor * multiplier * (actual_home - expected_home)
            new_r_home = r_home + delta
            new_r_away = r_away - delta

            self.ratings[g.home_team] = new_r_home
            self.ratings[g.away_team] = new_r_away

            rows.append(TeamGameResult(
                g.id, g.season, g.week, g.start_date, g.home_team, g.away_team, True,
                home_tier, away_tier, r_home, new_r_home, expected_home, home_won,
                g.home_points, g.away_points,
            ))
            rows.append(TeamGameResult(
                g.id, g.season, g.week, g.start_date, g.away_team, g.home_team, False,
                away_tier, home_tier, r_away, new_r_away, 1 - expected_home, not home_won,
                g.away_points, g.home_points,
            ))

        return pd.DataFrame(rows)

"""
Phase 4, steps 3-4: full-season Monte Carlo simulation.

Ratings evolve *within* each simulated season -- a team that gets lucky
early in one simulated universe is correctly favored more in its later
games in that same universe, and its opponents' ratings react too. That
rules out the easy "simulate each game independently off static preseason
ratings" shortcut, which would understate variance in the tails and ignore
how one game's result changes the next.

Running the chronological Elo loop from elo.py once per simulation (10,000+
sims x ~800 games) is slow in pure Python. Instead, the loop below iterates
over GAMES ONCE (fixed schedule order, ~800 iterations) and updates ALL
simulations' ratings simultaneously with numpy -- "which simulation" is
just a vectorized axis. Same update math as EloRatingSystem.run(), batched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb_elo.elo import EloConfig
from cfb_elo.spreads import SpreadModel


def simulate_season(
    schedule: pd.DataFrame,
    preseason_ratings: dict[str, float],
    elo_config: EloConfig,
    spread_model: SpreadModel,
    margin_sd: float,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Returns a (n_sims x n_teams) DataFrame of simulated win totals, one
    column per team that appears in `schedule`."""
    rng = np.random.default_rng(seed)
    schedule = schedule.sort_values(["week", "start_date"]).reset_index(drop=True)

    teams = sorted(preseason_ratings)
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    ratings = np.tile(np.array([preseason_ratings[t] for t in teams], dtype=float), (n_sims, 1))
    win_counts = np.zeros((n_sims, n_teams), dtype=np.int32)

    hfa = elo_config.home_field_advantage
    k = elo_config.k_factor
    mov_cap = elo_config.mov_cap

    missing = set()
    for g in schedule.itertuples(index=False):
        if g.home_team not in team_idx or g.away_team not in team_idx:
            missing.add((g.home_team, g.away_team))
            continue

        hi, ai = team_idx[g.home_team], team_idx[g.away_team]
        r_home = ratings[:, hi]
        r_away = ratings[:, ai]

        predicted_spread = spread_model.predict_spread(r_home - r_away)
        margin = rng.normal(loc=predicted_spread, scale=margin_sd, size=n_sims)
        home_win = margin > 0
        actual_home = home_win.astype(float)

        expected_home = 1.0 / (1.0 + 10 ** (-((r_home + hfa) - r_away) / 400.0))

        winner_rating = np.where(home_win, r_home, r_away)
        loser_rating = np.where(home_win, r_away, r_home)
        capped_margin = np.minimum(np.abs(margin), mov_cap)
        multiplier = np.log(capped_margin + 1) * (2.2 / (0.001 * (winner_rating - loser_rating) + 2.2))

        delta = k * multiplier * (actual_home - expected_home)
        ratings[:, hi] = r_home + delta
        ratings[:, ai] = r_away - delta

        win_counts[:, hi] += home_win
        win_counts[:, ai] += ~home_win

    if missing:
        raise KeyError(f"{len(missing)} schedule matchups have a team missing from preseason_ratings: "
                        f"{sorted(missing)[:5]}...")

    return pd.DataFrame(win_counts, columns=teams)


def over_under(win_totals: pd.Series, line: float) -> dict:
    """P(over)/P(under)/P(push) for a team's simulated win-total distribution
    against a posted line. `line` may be a .5 line (no push possible) or an
    integer line (push = exactly that many wins)."""
    n = len(win_totals)
    p_over = float((win_totals > line).mean())
    p_under = float((win_totals < line).mean())
    p_push = float((win_totals == line).mean()) if float(line).is_integer() else 0.0
    return {
        "line": line,
        "p_over": p_over,
        "p_under": p_under,
        "p_push": p_push,
        "mean_wins": float(win_totals.mean()),
        "median_wins": float(win_totals.median()),
        "n_sims": n,
    }

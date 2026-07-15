"""
Phase 4, steps 1-2: convert Elo rating differentials into a point spread,
then that spread into a win probability.

Both are fit directly against this project's own historical results (final
margins and outcomes) rather than real sportsbook closing lines -- we
haven't pulled betting-lines data yet. The intercept of the spread model
ends up being the average home-field edge in *real points*, estimated
straight from the data, which doubles as a sanity check on the Elo-space
home_field_advantage chosen in Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SpreadModel:
    intercept: float    # average home-field edge, in real points
    slope: float         # points of expected margin per Elo rating point of (home - away)
    residual_sd: float   # std dev of (actual margin - predicted spread) -- used as the Monte Carlo margin sd
    r_squared: float

    def predict_spread(self, rating_diff):
        return self.intercept + self.slope * rating_diff


@dataclass
class WinProbModel:
    intercept: float
    slope: float

    def predict(self, spread):
        z = self.intercept + self.slope * np.asarray(spread, dtype=float)
        return 1.0 / (1.0 + np.exp(-z))


def build_matchup_frame(history: pd.DataFrame) -> pd.DataFrame:
    """One row per FBS-vs-FBS game, with both teams' pre-game ratings, the
    actual margin, and the outcome. FCS-opponent games are excluded so they
    don't distort the fit -- they're close to guaranteed wins and would
    make the model look more confident than it should be for real matchups."""
    home = history[history["is_home"]]
    away = history[~history["is_home"]]
    merged = home.merge(away, on="game_id", suffixes=("_home", "_away"))
    fbs = merged[(merged["tier_home"] != "FCS") & (merged["tier_away"] != "FCS")].copy()

    fbs["rating_diff"] = fbs["pre_game_rating_home"] - fbs["pre_game_rating_away"]
    fbs["margin"] = fbs["points_for_home"] - fbs["points_for_away"]
    fbs["home_win"] = fbs["margin"] > 0
    return fbs


def fit_spread_model(matchups: pd.DataFrame) -> SpreadModel:
    x = matchups["rating_diff"].to_numpy()
    y = matchups["margin"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residuals = y - predicted

    residual_sd = float(np.std(residuals, ddof=2))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot

    return SpreadModel(intercept=float(intercept), slope=float(slope), residual_sd=residual_sd, r_squared=r_squared)


def _logistic_irls(x: np.ndarray, y: np.ndarray, n_iter: int = 25) -> tuple[float, float]:
    """Minimal 2-parameter (intercept, slope) logistic regression via
    iteratively reweighted least squares -- avoids adding a scikit-learn/
    statsmodels dependency for what's just a 1-feature fit."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(n_iter):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1 - p), 1e-6, None)
        hessian = (X.T * w) @ X
        gradient = X.T @ (y - p)
        beta = beta + np.linalg.solve(hessian, gradient)
    return float(beta[0]), float(beta[1])


def fit_win_prob_model(matchups: pd.DataFrame, spread_model: SpreadModel) -> WinProbModel:
    spread = spread_model.predict_spread(matchups["rating_diff"].to_numpy())
    y = matchups["home_win"].astype(float).to_numpy()
    intercept, slope = _logistic_irls(spread, y)
    return WinProbModel(intercept=intercept, slope=slope)

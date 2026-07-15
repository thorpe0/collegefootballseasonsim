"""
Phase 3: walk-forward backtesting and hyperparameter calibration.

Two layers of "no lookahead" are at work here:

1. Within a single Elo run, `EloRatingSystem.run()` processes games strictly
   chronologically -- a game's predicted win probability only ever depends
   on ratings built from *earlier* games. This is true for every config,
   always.

2. Across hyperparameter choices, `grid_search_walk_forward` uses an
   expanding-window season split: for each held-out season, the grid search
   only looks at Brier score on *prior* seasons before picking a K-factor /
   MOV cap / regression % combo, then reports that combo's performance on
   the held-out season it never used for selection. This mimics "what would
   I have picked, and how well would it have done, if I were running this
   system in real time" -- rather than fitting hyperparameters to the whole
   dataset and grading on the same data.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfb_elo.elo import EloConfig, EloRatingSystem

logger = logging.getLogger(__name__)


def brier_score(probs: pd.Series, outcomes: pd.Series) -> float:
    return float(((probs.to_numpy() - outcomes.to_numpy()) ** 2).mean())


def log_loss(probs: pd.Series, outcomes: pd.Series, eps: float = 1e-9) -> float:
    p = np.clip(probs.to_numpy(), eps, 1 - eps)
    y = outcomes.to_numpy()
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def home_rows(history: pd.DataFrame, fbs_only: bool = True) -> pd.DataFrame:
    """One row per game (the home team's perspective). `fbs_only` drops games
    involving an FCS opponent -- those are close to auto-wins and would make
    the Brier score look better than it really is for the matchups that
    actually matter for simulating an FBS season."""
    df = history[history["is_home"]]
    if fbs_only:
        df = df[(df["tier"] != "FCS") & (df["opponent_tier"] != "FCS")]
    return df


def evaluate_config(games: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    history = EloRatingSystem(cfg).run(games)
    return home_rows(history)


@dataclass
class FoldResult:
    test_season: int
    chosen_config: EloConfig
    train_brier: float
    test_brier: float
    baseline_brier: float


def _score_config(games: pd.DataFrame, cfg: EloConfig, exclude_first_season: bool) -> float:
    home = evaluate_config(games, cfg)
    if exclude_first_season:
        first = games["season"].min()
        scoped = home[home["season"] > first]
        if scoped.empty:
            scoped = home
    else:
        scoped = home
    return brier_score(scoped["win_probability"], scoped["won"].astype(float))


def grid_search_walk_forward(
    games: pd.DataFrame,
    k_values=(15, 20, 25, 30),
    mov_cap_values=(21, 28, 35),
    regression_values=(0.2, 0.35, 0.5),
    hfa_values=(45, 65, 85),
) -> list[FoldResult]:
    seasons = sorted(games["season"].unique())
    combos = list(itertools.product(k_values, mov_cap_values, regression_values, hfa_values))
    results: list[FoldResult] = []

    for test_season in seasons[1:]:
        train_games = games[games["season"] < test_season]
        full_games = games[games["season"] <= test_season]

        best_score, best_cfg = None, None
        for k, mov_cap, reg, hfa in combos:
            cfg = EloConfig(k_factor=k, mov_cap=mov_cap, regression_pct=reg, home_field_advantage=hfa)
            score = _score_config(train_games, cfg, exclude_first_season=True)
            if best_score is None or score < best_score:
                best_score, best_cfg = score, cfg

        full_home = evaluate_config(full_games, best_cfg)
        test_home = full_home[full_home["season"] == test_season]
        test_brier = brier_score(test_home["win_probability"], test_home["won"].astype(float))
        baseline_brier = brier_score(pd.Series(0.5, index=test_home.index), test_home["won"].astype(float))

        result = FoldResult(test_season, best_cfg, best_score, test_brier, baseline_brier)
        results.append(result)
        logger.info(
            "Test season %s: chosen K=%.0f mov_cap=%.0f reg=%.2f hfa=%.0f | "
            "train Brier=%.4f  test Brier=%.4f  (0.5-baseline=%.4f)",
            test_season, best_cfg.k_factor, best_cfg.mov_cap, best_cfg.regression_pct,
            best_cfg.home_field_advantage, best_score, test_brier, baseline_brier,
        )

    return results


def select_final_config(
    games: pd.DataFrame,
    k_values=(15, 20, 25, 30),
    mov_cap_values=(21, 28, 35),
    regression_values=(0.2, 0.35, 0.5),
    hfa_values=(45, 65, 85),
) -> tuple[EloConfig, float]:
    """Grid search over ALL available seasons -- used once walk-forward
    validation shows the tuning approach generalizes, to produce the config
    actually used going forward (Phase 4)."""
    combos = list(itertools.product(k_values, mov_cap_values, regression_values, hfa_values))
    best_score, best_cfg = None, None
    for k, mov_cap, reg, hfa in combos:
        cfg = EloConfig(k_factor=k, mov_cap=mov_cap, regression_pct=reg, home_field_advantage=hfa)
        score = _score_config(games, cfg, exclude_first_season=True)
        if best_score is None or score < best_score:
            best_score, best_cfg = score, cfg
    return best_cfg, best_score


def calibration_table(history_home: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted win-probability bucket vs. actual win rate."""
    df = history_home.copy()
    df["bucket"] = pd.cut(df["win_probability"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    table = df.groupby("bucket", observed=True).agg(
        n_games=("won", "size"),
        avg_predicted=("win_probability", "mean"),
        actual_win_rate=("won", "mean"),
    )
    return table.reset_index()

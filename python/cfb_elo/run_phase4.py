"""
Phase 4: point-spread conversion + full-season Monte Carlo simulation.

Usage:
    python -m cfb_elo.run_phase4
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from cfb_elo import config
from cfb_elo.elo import EloConfig, EloRatingSystem, classify_team_tier
from cfb_elo.schedule import fetch_schedule
from cfb_elo.simulate import over_under, simulate_season
from cfb_elo.spreads import build_matchup_frame, fit_spread_model, fit_win_prob_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_SIMS = 20_000


def load_tuned_config() -> EloConfig:
    cfg_path = config.DATA_PROCESSED_DIR / "elo_config.json"
    return EloConfig(**json.loads(cfg_path.read_text()))


def team_info_from_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per team appearing in the schedule, with its conference and
    classification for the target season."""
    home = schedule[["home_team", "home_conference", "home_classification"]].rename(
        columns={"home_team": "team", "home_conference": "conference", "home_classification": "classification"}
    )
    away = schedule[["away_team", "away_conference", "away_classification"]].rename(
        columns={"away_team": "team", "away_conference": "conference", "away_classification": "classification"}
    )
    return pd.concat([home, away]).drop_duplicates("team").set_index("team")


def main() -> None:
    elo_config = load_tuned_config()
    logger.info("Using tuned config: %s", elo_config)

    games_path = config.DATA_PROCESSED_DIR / f"games_{config.START_SEASON}_{config.END_SEASON}.parquet"
    games = pd.read_parquet(games_path)

    logger.info("Running historical Elo through %s to build spread/win-prob models and 2026 preseason state...",
                config.END_SEASON)
    elo = EloRatingSystem(elo_config)
    history = elo.run(games)

    matchups = build_matchup_frame(history)
    spread_model = fit_spread_model(matchups)
    win_prob_model = fit_win_prob_model(matchups, spread_model)

    print("\n=== Spread model: margin ~ intercept + slope * rating_diff (OLS on actual results) ===")
    print(f"  intercept (home-field edge, real points): {spread_model.intercept:.2f}")
    print(f"  slope (points of margin per Elo rating point):   {spread_model.slope:.4f}")
    print(f"  residual sd (-> Monte Carlo margin sd):          {spread_model.residual_sd:.2f}  "
          f"(spec's rule-of-thumb prior was 16-17)")
    print(f"  R^2:                                             {spread_model.r_squared:.3f}")
    implied_hfa_pts = elo_config.home_field_advantage * spread_model.slope
    print(f"  cross-check: configured Elo-space HFA ({elo_config.home_field_advantage} pts) "
          f"* slope = {implied_hfa_pts:.2f} real points (vs. {spread_model.intercept:.2f} fitted directly)")

    print("\n=== Win-probability model: P(home win) = logistic(intercept + slope * spread) ===")
    print(f"  intercept: {win_prob_model.intercept:.4f}  slope: {win_prob_model.slope:.4f}")
    for s in [-21, -14, -7, -3, 0, 3, 7, 14, 21]:
        print(f"  spread={s:+4d} -> P(home win) = {float(win_prob_model.predict(s)):.3f}")

    logger.info("Fetching %s schedule...", config.TARGET_SEASON)
    schedule = fetch_schedule(config.TARGET_SEASON)
    stem = f"schedule_{config.TARGET_SEASON}"
    schedule.to_parquet(config.DATA_PROCESSED_DIR / f"{stem}.parquet", index=False)
    schedule.to_csv(config.DATA_PROCESSED_DIR / f"{stem}.csv", index=False)

    team_info = team_info_from_schedule(schedule)
    preseason_ratings = {}
    for team, row in team_info.iterrows():
        tier = classify_team_tier(team, row["conference"], row["classification"])
        preseason_ratings[team] = elo.preseason_rating(team, config.TARGET_SEASON, tier)

    logger.info("Running %d full-season Monte Carlo simulations over %d scheduled games...",
                N_SIMS, len(schedule))
    win_totals = simulate_season(
        schedule=schedule,
        preseason_ratings=preseason_ratings,
        elo_config=elo_config,
        spread_model=spread_model,
        margin_sd=spread_model.residual_sd,
        n_sims=N_SIMS,
        seed=42,
    )

    out_stem = f"sim_{config.TARGET_SEASON}_win_totals"
    win_totals.to_parquet(config.DATA_PROCESSED_DIR / f"{out_stem}.parquet", index=False)
    win_totals.to_csv(config.DATA_PROCESSED_DIR / f"{out_stem}.csv", index=False)
    logger.info("Wrote %d simulations x %d teams to %s.parquet / .csv", N_SIMS, win_totals.shape[1], out_stem)

    fbs_teams = team_info[team_info["classification"] == "fbs"].index
    summary = pd.DataFrame({
        "mean_wins": win_totals[fbs_teams].mean(),
        "median_wins": win_totals[fbs_teams].median(),
        "std_wins": win_totals[fbs_teams].std(),
        "p10": win_totals[fbs_teams].quantile(0.10),
        "p90": win_totals[fbs_teams].quantile(0.90),
        # .reindex, not a bare Series -- preseason_ratings covers all 238
        # teams (FCS opponents included), while the columns above only
        # cover the 138 FBS teams; mixing a full-index Series into this
        # dict would silently union-join in ~100 NaN-filled FCS rows.
        "preseason_rating": pd.Series(preseason_ratings).reindex(fbs_teams),
    }).sort_values("mean_wins", ascending=False)
    summary["preseason_rating"] = summary["preseason_rating"].round(1)
    summary = summary.round(2)

    summary_path = config.DATA_PROCESSED_DIR / f"sim_{config.TARGET_SEASON}_summary.csv"
    summary.to_csv(summary_path)
    logger.info("Wrote per-team summary to %s", summary_path)

    print(f"\nTop 15 teams by projected mean {config.TARGET_SEASON} regular-season wins:")
    print(summary.head(15).to_string())
    print(f"\nBottom 10 teams by projected mean {config.TARGET_SEASON} regular-season wins:")
    print(summary.tail(10).to_string())

    print("\n=== Example over/under output (illustrative lines only -- not real posted lines) ===")
    example_teams = summary.head(8).index[::2].tolist() + summary.index[len(summary) // 2:len(summary) // 2 + 2].tolist()
    for team in example_teams:
        line = round(summary.loc[team, "mean_wins"] * 2) / 2  # nearest .5, mimicking a book's line near the model's mean
        result = over_under(win_totals[team], line)
        print(f"  {team:<20} line={line:>4}  P(over)={result['p_over']:.1%}  "
              f"P(under)={result['p_under']:.1%}  mean={result['mean_wins']:.2f}")


if __name__ == "__main__":
    main()

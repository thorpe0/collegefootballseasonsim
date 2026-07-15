# CFB Elo Betting Model

Simplified Elo-style rating system for predicting CFB season win totals,
built in phases:

1. **Data pull** — historical FBS game results from CollegeFootballData.com
2. **Elo rating engine** — MOV-adjusted Elo with home-field edge and preseason regression
3. **Backtesting/calibration** — walk-forward tuning against Brier score
4. **Season Monte Carlo simulation** — win-total distributions vs. sportsbook lines

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your free API key from
https://collegefootballdata.com/key:

```
CFBD_API_KEY=your_key_here
```

## Phase 1: pull game data

```
python -m cfb_elo.data_pull
```

This pulls every game involving an FBS team (regular season + postseason)
for the seasons configured in `cfb_elo/config.py` (default: 2021–2025),
and writes:

- `data/raw/games_<season>.parquet` — one file per season, as returned by the API
- `data/processed/games_2021_2025.parquet` — the cleaned, combined dataset used downstream

### Columns

| Column | Description |
|---|---|
| `id` | CFBD game ID |
| `season`, `week`, `season_type` | When the game was played (`season_type`: regular/postseason) |
| `start_date` | Kickoff timestamp (UTC) |
| `neutral_site`, `conference_game` | Game context flags |
| `home_team` / `away_team` | Team names |
| `home_conference` / `away_conference` | Conference at time of game |
| `home_classification` / `away_classification` | `fbs`, `fcs`, `ii`, `iii` |
| `home_points` / `away_points` | Final score |
| `is_fcs_game` | True if either side is not FBS (kept, not dropped, per project spec) |
| `margin` | `home_points - away_points` |
| `home_win` | `margin > 0` |

Games without a final score (not yet played) are dropped automatically.

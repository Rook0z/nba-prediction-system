from nba_api.stats.endpoints import scoreboardv3, boxscoretraditionalv3
from nba_api.stats.static import teams
from datetime import datetime
import pandas as pd
import unicodedata
import time


# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_FILE = "./results/todays_actuals.csv"

PLAYER_LIST = [
    "Shai Gilgeous-Alexander",
    "Giannis Antetokounmpo",
    "James Harden",
    "Kawhi Leonard",
    "Stephen Curry",
    "Damian Lillard",
    "Luka Dončić",
    "Nikola Jokić",
    "Kevin Durant",
    "LeBron James",
    "Anthony Davis",
    "Jayson Tatum",
    "Trae Young",
    "Devin Booker",
    "Bam Adebayo",
    "Chet Holmgren",
    "Rudy Gobert",
    "Jalen Brunson",
    "Julius Randle",
    "Jaylen Brown",
    "Donovan Mitchell",
    "Tyrese Haliburton",
]


# -----------------------------
# HELPERS
# -----------------------------
def normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    return name.lower().strip()


NORMALIZED_PLAYERS = {normalize(p): p for p in PLAYER_LIST}


def get_team_abbr(team_id):
    for t in teams.get_teams():
        if t["id"] == team_id:
            return t["abbreviation"]
    return ""


# -----------------------------
# GAME FETCH
# -----------------------------
def get_valid_game_ids():
    today = datetime.now().strftime("%Y-%m-%d")
    board = scoreboardv3.ScoreboardV3(game_date=today)
    games = board.game_header.get_data_frame()

    valid = games[
        games["gameStatus"].isin([2, 3])  # 2 = Live, 3 = Final
    ]

    return valid["gameId"].tolist()


# -----------------------------
# STATS FETCH
# -----------------------------
def fetch_stats():
    game_ids = get_valid_game_ids()
    results = {}

    if not game_ids:
        return results

    for game_id in game_ids:
        time.sleep(0.7)

        try:
            box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
            df = box.player_stats.get_data_frame()

            if df is None or df.empty:
                continue

        except Exception:
            # NBA API sometimes returns broken payloads
            continue

        for _, r in df.iterrows():
            api_name = r.get("playerName")
            if not api_name:
                continue

            norm_name = normalize(api_name)

            if norm_name not in NORMALIZED_PLAYERS:
                continue

            results[norm_name] = {
                "Player": NORMALIZED_PLAYERS[norm_name],
                "Team": get_team_abbr(r.get("teamId")),
                "PTS_ACTUAL": int(r.get("points", 0)),
                "REB_ACTUAL": int(r.get("reboundsTotal", 0)),
                "AST_ACTUAL": int(r.get("assists", 0)),
                "DID_PLAY": "YES",
            }

    return results


# -----------------------------
# MAIN
# -----------------------------
def main():
    stats = fetch_stats()

    rows = []

    for norm_name, original_name in NORMALIZED_PLAYERS.items():
        if norm_name in stats:
            rows.append(stats[norm_name])
        else:
            rows.append({
                "Player": original_name,
                "Team": "",
                "PTS_ACTUAL": "",
                "REB_ACTUAL": "",
                "AST_ACTUAL": "",
                "DID_PLAY": "NO",
            })

    df = pd.DataFrame(rows)
    df = df[
        ["Player", "Team", "PTS_ACTUAL", "REB_ACTUAL", "AST_ACTUAL", "DID_PLAY"]
    ]

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✓ Saved today's actuals to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

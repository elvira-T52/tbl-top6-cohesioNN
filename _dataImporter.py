import os
import time

import pandas as pd

from _NHL_game import NHL_game, fetch_team_games, DEFAULT_FEATURE_COLS


GAME_ID = 2024020020
TEAM = "TBL"

PLAYERS = {
    "Brandon Hagel": 8479542,
    "Brayden Point": 8478010,
    "Nikita Kucherov": 8476453,
    "Victor Hedman": 8475167,
    # traded to TBL in March 2025 -- earlier seasons will correctly yield
    # ~0 sequences for him via player_is_on_team, not an error
    "Jake Guentzel": 8477404,
    "Anthony Cirelli": 8478519,
}
PLAYER_ID = PLAYERS["Brandon Hagel"]  # used by the single-game smoke test below


def season_str(start_year):
    # NHL season strings are "YYYYYYYY", e.g. the 2022-23 season is "20222023".
    return f"{start_year}{start_year + 1}"


SEASON_START_YEARS = range(2022, 2026)  # 2022-23 through 2025-26
SEASONS = [season_str(y) for y in SEASON_START_YEARS]

DATA_DIR = os.path.dirname(__file__)
COMBINED_CACHE_PATH = os.path.join(DATA_DIR, "tbl_all_players_sequences.csv")


def cache_path_for_player_season(player_id, season):
    return os.path.join(DATA_DIR, f"sequences_{player_id}_{season}.csv")


def read_csv_or_empty(path):
    # A player with 0 qualifying games in a season (e.g. Guentzel pre-trade)
    # produces a genuinely empty (0-row, 0-column) cache file, which
    # pd.read_csv rejects with EmptyDataError -- treat that as "0 sequences"
    # instead of crashing.
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def sequences_to_dataframe(sequences, labels, game_ids, feature_cols=DEFAULT_FEATURE_COLS):
    # Wide format: one row per training example (one per sequence), with a
    # column per (timestep, feature) plus the label -- easy to load back with
    # pd.read_csv and easy to encode column-by-column later.
    rows = []
    for seq, label, game_id in zip(sequences, labels, game_ids):
        row = {"game_id": game_id}
        for t, timestep in enumerate(seq):
            for col, val in zip(feature_cols, timestep):
                row[f"t{t}_{col}"] = val
        row["label_actorId"] = label["actorId"]
        row["label_actorName"] = label["actorName"]
        row["label_typeDescKey"] = label["typeDescKey"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_season_sequences_multi_player(team_abbrev, season, players, k=5, delay_sec=0.3):
    # Fetches each game in the season exactly ONCE, then builds sequences for
    # every player against that same cached game -- avoids re-fetching the
    # same play-by-play/shift-chart data once per player (6x fewer requests
    # than looping players on the outside and games on the inside).
    game_ids = fetch_team_games(team_abbrev, season)
    results = {name: {"sequences": [], "labels": [], "game_ids": []} for name in players}

    for i, game_id in enumerate(game_ids):
        print(f"[{i + 1}/{len(game_ids)}] game {game_id}")
        try:
            game = NHL_game(game_id)
            for name, player_id in players.items():
                # skip players who weren't rostered for this team in THIS
                # game -- guards against counting a traded player's plays for
                # their old team (see player_is_on_team's docstring)
                if not game.player_is_on_team(player_id, team_abbrev):
                    continue
                sequences, labels = game.build_line_actor_sequences(player_id, k=k)
                results[name]["sequences"].extend(sequences)
                results[name]["labels"].extend(labels)
                results[name]["game_ids"].extend([game_id] * len(sequences))
        except Exception as e:
            print("  skipped:", e)
        time.sleep(delay_sec)  # be polite to an undocumented API

    return results


def load_or_build_multi_player_season(team_abbrev, season, players, k=5):
    # Loads each player's cached CSV for this season where it already exists;
    # only fetches the season's games for whichever players are still missing.
    cached = {}
    missing = {}
    for name, player_id in players.items():
        path = cache_path_for_player_season(player_id, season)
        if os.path.exists(path):
            cached[name] = read_csv_or_empty(path)
        else:
            missing[name] = player_id

    if missing:
        print(f"fetching {season} for: {list(missing.keys())}")
        results = build_season_sequences_multi_player(team_abbrev, season, missing, k=k)
        for name, player_id in missing.items():
            data = results[name]
            df = sequences_to_dataframe(data["sequences"], data["labels"], data["game_ids"])
            path = cache_path_for_player_season(player_id, season)
            df.to_csv(path, index=False)
            print(f"saved {len(df)} sequences to {path}")
            cached[name] = df

    return cached


def load_or_build_multi_player_dataset(team_abbrev, players, seasons, k=5,
                                        combined_cache_path=COMBINED_CACHE_PATH):
    if os.path.exists(combined_cache_path):
        print(f"loading combined cache from {combined_cache_path}")
        return pd.read_csv(combined_cache_path)

    all_dfs = []
    for season in seasons:
        season_dfs = load_or_build_multi_player_season(team_abbrev, season, players, k=k)
        for name, df in season_dfs.items():
            if len(df) == 0:
                continue  # e.g. Guentzel's pre-trade seasons
            df = df.copy()
            df["season"] = season
            df["player_name"] = name
            df["player_id"] = players[name]
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(combined_cache_path, index=False)
    print(f"saved {len(combined)} total sequences to {combined_cache_path}")
    return combined


SHOT_COMBINED_CACHE_PATH = os.path.join(DATA_DIR, "tbl_shot_outcomes.csv")


def shot_cache_path_for_player_season(player_id, season):
    return os.path.join(DATA_DIR, f"shots_{player_id}_{season}.csv")


def shot_sequences_to_dataframe(sequences, labels, game_ids, feature_cols=DEFAULT_FEATURE_COLS):
    # Same wide format as sequences_to_dataframe, but each sequence is k+1
    # timesteps (the shot itself is the last one) and the label is a single
    # 0/1 -- was this shot a goal -- instead of a who/what dict.
    rows = []
    for seq, label, game_id in zip(sequences, labels, game_ids):
        row = {"game_id": game_id, "label_isGoal": label}
        for t, timestep in enumerate(seq):
            for col, val in zip(feature_cols, timestep):
                row[f"t{t}_{col}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def build_shot_season_multi_player(team_abbrev, season, players, k=5, delay_sec=0.3):
    # Same one-fetch-per-game structure as build_season_sequences_multi_player,
    # just calling build_shot_outcome_sequences instead.
    game_ids = fetch_team_games(team_abbrev, season)
    results = {name: {"sequences": [], "labels": [], "game_ids": []} for name in players}

    for i, game_id in enumerate(game_ids):
        print(f"[{i + 1}/{len(game_ids)}] game {game_id}")
        try:
            game = NHL_game(game_id)
            for name, player_id in players.items():
                if not game.player_is_on_team(player_id, team_abbrev):
                    continue
                sequences, labels = game.build_shot_outcome_sequences(player_id, k=k)
                results[name]["sequences"].extend(sequences)
                results[name]["labels"].extend(labels)
                results[name]["game_ids"].extend([game_id] * len(sequences))
        except Exception as e:
            print("  skipped:", e)
        time.sleep(delay_sec)  # be polite to an undocumented API

    return results


def load_or_build_shot_season(team_abbrev, season, players, k=5):
    cached = {}
    missing = {}
    for name, player_id in players.items():
        path = shot_cache_path_for_player_season(player_id, season)
        if os.path.exists(path):
            cached[name] = read_csv_or_empty(path)
        else:
            missing[name] = player_id

    if missing:
        print(f"fetching shot outcomes for {season}: {list(missing.keys())}")
        results = build_shot_season_multi_player(team_abbrev, season, missing, k=k)
        for name, player_id in missing.items():
            data = results[name]
            df = shot_sequences_to_dataframe(data["sequences"], data["labels"], data["game_ids"])
            path = shot_cache_path_for_player_season(player_id, season)
            df.to_csv(path, index=False)
            print(f"saved {len(df)} shot sequences to {path}")
            cached[name] = df

    return cached


def load_or_build_shot_dataset(team_abbrev, players, seasons, k=5,
                                combined_cache_path=SHOT_COMBINED_CACHE_PATH):
    if os.path.exists(combined_cache_path):
        print(f"loading combined cache from {combined_cache_path}")
        return pd.read_csv(combined_cache_path)

    all_dfs = []
    for season in seasons:
        season_dfs = load_or_build_shot_season(team_abbrev, season, players, k=k)
        for name, df in season_dfs.items():
            if len(df) == 0:
                continue  # e.g. Guentzel's pre-trade seasons
            df = df.copy()
            df["season"] = season
            df["player_name"] = name
            df["player_id"] = players[name]
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(combined_cache_path, index=False)
    print(f"saved {len(combined)} total shot sequences to {combined_cache_path}")
    return combined


def main():
    game = NHL_game(GAME_ID)
    line = game.line_report(PLAYER_ID)
    print(len(line), "Hagel shifts")

    print("\n=== all plays made by Hagel's line this game ===")
    actions = game.line_actions(PLAYER_ID)
    print(actions[["period", "timeInPeriod", "typeDescKey", "zoneCode", "shotType", "shotAngle",
                    "teamStrength", "opponent", "scoreDifferential", "elapsedSeconds",
                    "Player1", "Player2", "Player3", "Player4", "Player5", "Player6"]])

    print(f"\n=== loading/building sequences for {list(PLAYERS.keys())} across {SEASONS} ===")
    all_df = load_or_build_multi_player_dataset(TEAM, PLAYERS, SEASONS, k=5)
    print(all_df.shape)
    print(all_df.groupby("player_name")["season"].value_counts())

    print(f"\n=== loading/building shot outcomes for {list(PLAYERS.keys())} across {SEASONS} ===")
    shot_df = load_or_build_shot_dataset(TEAM, PLAYERS, SEASONS, k=5)
    print(shot_df.shape)
    print(shot_df.groupby("player_name")["label_isGoal"].mean())  # goal rate per player


if __name__ == "__main__":
    main()

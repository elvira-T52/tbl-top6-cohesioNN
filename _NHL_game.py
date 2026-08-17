import math

import requests
import pandas as pd


HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_TIMEOUT = 10  # seconds -- without this, a stalled request hangs forever

# NHL's play-by-play "details" dict uses a different field name for "who did this"
# depending on the event type (e.g. a goal's scorer is "scoringPlayerId", a hit's
# hitter is "hittingPlayerId"). This maps event type -> the field to read.
ACTOR_FIELD_BY_EVENT = {
    "shot-on-goal": "shootingPlayerId",
    "missed-shot": "shootingPlayerId",
    "blocked-shot": "shootingPlayerId",
    "goal": "scoringPlayerId",
    "hit": "hittingPlayerId",
    "giveaway": "playerId",
    "takeaway": "playerId",
    "faceoff": "winningPlayerId",
}

SHOT_EVENT_TYPES = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}

# Stands in for a shot's own typeDescKey in build_shot_outcome_sequences, so
# the model can't just read "goal" straight off the input to get the answer.
SHOT_OUTCOME_PLACEHOLDER = "shot-attempt"

DEFAULT_FEATURE_COLS = [
    "typeDescKey", "zoneCode", "xCoord", "yCoord", "shotType", "shotAngle", "shotDistance",
    "teamStrength", "opponent", "scoreDifferential", "elapsedSeconds",
    "Player1", "Player2", "Player3", "Player4", "Player5", "Player6",
]

# NHL rink coordinates run roughly -100..100 on x, -42.5..42.5 on y, with the
# goal line (and net) sitting at about x = +-89 regardless of which end a team
# is attacking -- so using abs(xCoord) sidesteps needing to know which side.
GOAL_LINE_X = 89


def shot_angle(x_coord, y_coord):
    # Angle (degrees) between the shot location and the center of the goal:
    # 0 = dead-on/straight in front of the net, 90 = along the goal line.
    if x_coord is None or y_coord is None:
        return None
    adjacent = GOAL_LINE_X - abs(x_coord)  # straight-line distance toward the net
    opposite = abs(y_coord)
    return math.degrees(math.atan2(opposite, adjacent))


def shot_distance(x_coord, y_coord):
    # Straight-line distance (rink-coordinate units) from the shot location to
    # the center of the goal -- same right triangle shot_angle() uses.
    if x_coord is None or y_coord is None:
        return None
    adjacent = GOAL_LINE_X - abs(x_coord)
    opposite = abs(y_coord)
    return math.hypot(adjacent, opposite)


def fetch_team_games(team_abbrev, season, headers=HEADERS):
    # All completed regular-season game IDs for a team in a given season,
    # e.g. fetch_team_games("TBL", "20242025").
    resp = requests.get(
        f"https://api-web.nhle.com/v1/club-schedule-season/{team_abbrev}/{season}",
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    games = resp.json()["games"]
    return [g["id"] for g in games if g["gameState"] == "OFF" and g["gameType"] == 2]


def time_to_seconds(t):
    # NHL times come back as "MM:SS" strings (e.g. timeInPeriod, shift startTime/endTime).
    # Converting to seconds makes them comparable/sortable.
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def elapsed_seconds(period, time_in_period):
    # Collapses (period, timeInPeriod) into one continuous "seconds into the
    # game" value -- regulation periods are 20 minutes each. Simpler for a
    # model to consume than two separate period/time-remaining features.
    return (period - 1) * 1200 + time_to_seconds(time_in_period)


def _flatten_play(play, players):
    # Turns one raw play-by-play event (nested dict) into a single flat dict,
    # ready to become one row of a DataFrame.
    details = play.get("details", {})
    actor_field = ACTOR_FIELD_BY_EVENT.get(play["typeDescKey"])
    actor_id = details.get(actor_field) if actor_field else None

    return {
        "eventId": play["eventId"],
        "period": play["periodDescriptor"]["number"],
        "timeInPeriod": play["timeInPeriod"],
        "timeRemaining": play["timeRemaining"],
        "situationCode": play.get("situationCode"),  # 4-digit strength code, see parse_situation_code
        "typeDescKey": play["typeDescKey"],  # event type, e.g. "goal", "hit", "shot-on-goal"
        "sortOrder": play["sortOrder"],  # NHL's own chronological ordering of events
        "xCoord": details.get("xCoord"),
        "yCoord": details.get("yCoord"),
        "zoneCode": details.get("zoneCode"),
        "eventOwnerTeamId": details.get("eventOwnerTeamId"),
        # only present on shot/goal events
        "shotType": details.get("shotType"),
        "shotAngle": (
            shot_angle(details.get("xCoord"), details.get("yCoord"))
            if play["typeDescKey"] in SHOT_EVENT_TYPES else None
        ),
        "shotDistance": (
            shot_distance(details.get("xCoord"), details.get("yCoord"))
            if play["typeDescKey"] in SHOT_EVENT_TYPES else None
        ),
        "awayScore": details.get("awayScore"),  # only present on "goal" events
        "homeScore": details.get("homeScore"),  # only present on "goal" events
        "actorId": actor_id,
        "actorName": players.get(actor_id),
        # only present on "goal" events
        "assist1Id": details.get("assist1PlayerId"),
        "assist1Name": players.get(details.get("assist1PlayerId")),
        "assist2Id": details.get("assist2PlayerId"),
        "assist2Name": players.get(details.get("assist2PlayerId")),
    }


def flatten_plays(plays, players):
    # Flattens every play in a game into one DataFrame, ordered by when they happened.
    rows = [_flatten_play(p, players) for p in plays]
    return pd.DataFrame(rows).sort_values("sortOrder").reset_index(drop=True)


def parse_situation_code(code):
    # situationCode is a 4-digit string: [awayGoalieIn][awaySkaters][homeSkaters][homeGoalieIn]
    # e.g. "1551" = both goalies in, 5 skaters each side = even strength (5v5).
    # "1451" = away has only 4 skaters = away shorthanded / home power play.
    if code is None or len(str(code)) != 4:
        return pd.Series({
            "awayGoalieIn": None, "awaySkaters": None,
            "homeSkaters": None, "homeGoalieIn": None,
            "strengthState": None,
        })
    away_goalie, away_skaters, home_skaters, home_goalie = (int(c) for c in str(code))
    return pd.Series({
        "awayGoalieIn": bool(away_goalie),
        "awaySkaters": away_skaters,
        "homeSkaters": home_skaters,
        "homeGoalieIn": bool(home_goalie),
        "strengthState": f"{away_skaters}v{home_skaters}",
    })


class NHL_game:
    # Wraps one NHL game: fetches play-by-play + shift chart data from the NHL API,
    # caches it, and exposes it as a ready-to-use DataFrame plus shift/on-ice helpers.

    def __init__(self, game_id, headers=HEADERS):
        self.game_id = game_id
        self.headers = headers
        self._pbp_data = None  # cache: raw play-by-play JSON
        self._shifts = None    # cache: raw shift chart data
        self._df = None        # cache: flattened/parsed DataFrame

    def fetch_play_by_play(self):
        # Raw play-by-play JSON for this game (plays, rosterSpots, teams, etc.).
        if self._pbp_data is None:  # only hit the API once
            resp = requests.get(
                f"https://api-web.nhle.com/v1/gamecenter/{self.game_id}/play-by-play",
                headers=self.headers,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            self._pbp_data = resp.json()
        return self._pbp_data

    def fetch_shift_chart(self):
        # Raw shift chart: one record per player per shift (start/end time, period, team).
        if self._shifts is None:
            resp = requests.get(
                "https://api.nhle.com/stats/rest/en/shiftcharts",
                params={"cayenneExp": f"gameId={self.game_id}"},
                headers=self.headers,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            self._shifts = resp.json()["data"]
        return self._shifts

    @property
    def players(self):
        # {playerId: "First Last"} lookup built from this game's roster.
        roster_spots = self.fetch_play_by_play()["rosterSpots"]
        return {
            p["playerId"]: f"{p['firstName']['default']} {p['lastName']['default']}"
            for p in roster_spots
        }

    @property
    def goalie_ids(self):
        # Player IDs on either roster whose position is goaltender.
        # (Shift chart records don't carry position, so we cross-reference rosterSpots.)
        roster_spots = self.fetch_play_by_play()["rosterSpots"]
        return {p["playerId"] for p in roster_spots if p["positionCode"] == "G"}

    @property
    def dataframe(self):
        # Fully flattened + strength-parsed play-by-play for this game, one row per event.
        if self._df is None:
            data = self.fetch_play_by_play()
            df = flatten_plays(data["plays"], self.players)
            df = pd.concat([df, df["situationCode"].apply(parse_situation_code)], axis=1)
            # awayScore/homeScore only arrive on "goal" rows -- forward-fill so every
            # row carries "the score as of this point in the game," starting at 0-0.
            df["awayScore"] = df["awayScore"].ffill().fillna(0)
            df["homeScore"] = df["homeScore"].ffill().fillna(0)
            self._df = df
        return self._df

    def players_on_ice(self, period, start_sec, end_sec, exclude_player_id=None):
        # Every player (either team, including goalies) whose shift overlaps
        # the given [start_sec, end_sec) window in the given period.
        on_ice = []
        for s in self.fetch_shift_chart():
            if s["period"] != period:
                continue
            if exclude_player_id is not None and s["playerId"] == exclude_player_id:
                continue
            s_start = time_to_seconds(s["startTime"])
            s_end = time_to_seconds(s["endTime"])
            if s_start < end_sec and start_sec < s_end:  # interval overlap check
                on_ice.append(s)
        return on_ice

    def plays_during_shift(self, shift):
        # All play-by-play rows that happened during a given shift record.
        start_sec = time_to_seconds(shift["startTime"])
        end_sec = time_to_seconds(shift["endTime"])
        time_sec = self.dataframe["timeInPeriod"].apply(time_to_seconds)
        mask = (self.dataframe["period"] == shift["period"]) & time_sec.between(start_sec, end_sec)
        return self.dataframe[mask]

    def player_shifts(self, player_id):
        # All shift records belonging to one player, across the whole game.
        return [s for s in self.fetch_shift_chart() if s["playerId"] == player_id]

    def team_id_of(self, player_id):
        # Which team (by teamId) this player belongs to in this game.
        return next(s["teamId"] for s in self.fetch_shift_chart() if s["playerId"] == player_id)

    def player_team_id_for_game(self, player_id):
        # Same as team_id_of, but returns None instead of raising when the
        # player didn't appear in this game at all (scratched, injured, etc.)
        # -- safe to call speculatively before knowing whether they played.
        return next(
            (s["teamId"] for s in self.fetch_shift_chart() if s["playerId"] == player_id), None
        )

    def player_is_on_team(self, player_id, team_abbrev):
        # Whether this player was actually rostered for `team_abbrev`
        # specifically IN THIS GAME. Guards against a traded/waived player's
        # plays getting counted for their old team just because a game
        # involving their new team happened to also involve their old one
        # (e.g. player was on the opponent's roster before being traded here).
        team_id = self.player_team_id_for_game(player_id)
        if team_id is None:
            return False
        data = self.fetch_play_by_play()
        target_id = (
            data["homeTeam"]["id"] if data["homeTeam"]["abbrev"] == team_abbrev
            else data["awayTeam"]["id"]
        )
        return team_id == target_id

    @property
    def home_team_id(self):
        return self.fetch_play_by_play()["homeTeam"]["id"]

    @property
    def away_team_id(self):
        return self.fetch_play_by_play()["awayTeam"]["id"]

    def opponent_abbrev(self, player_id):
        # This player's opposing team's abbreviation for this game.
        data = self.fetch_play_by_play()
        if self.team_id_of(player_id) == self.home_team_id:
            return data["awayTeam"]["abbrev"]
        return data["homeTeam"]["abbrev"]

    def team_strength(self, team_id, away_skaters, home_skaters):
        # From team_id's own perspective: "PP" (more skaters on ice than the
        # opponent), "PK" (fewer), or "EVEN" (same, e.g. 5v5, 4v4, 3v3).
        own, opp = (
            (home_skaters, away_skaters) if team_id == self.home_team_id
            else (away_skaters, home_skaters)
        )
        if own > opp:
            return "PP"
        if own < opp:
            return "PK"
        return "EVEN"

    def score_differential(self, team_id, away_score, home_score):
        # From team_id's own perspective: positive = leading, negative = trailing.
        own, opp = (
            (home_score, away_score) if team_id == self.home_team_id
            else (away_score, home_score)
        )
        return own - opp

    def _teammate_shifts(self, player_id, period):
        # This player's own team's skater shift records (goalie excluded, player
        # excluded) for a given period -- the candidate pool for "who was on his line."
        team_id = self.team_id_of(player_id)
        return [
            s for s in self.fetch_shift_chart()
            if s["period"] == period
            and s["teamId"] == team_id
            and s["playerId"] != player_id
            and s["playerId"] not in self.goalie_ids
        ]

    def teammates_on_ice(self, player_id, period, time_sec):
        # Teammates on the ice at one instant in time.
        names = []
        for s in self._teammate_shifts(player_id, period):
            if time_to_seconds(s["startTime"]) <= time_sec <= time_to_seconds(s["endTime"]):
                names.append(f"{s['firstName']} {s['lastName']}")
        return names

    def teammates_on_ice_during(self, player_id, period, start_sec, end_sec):
        # Teammate shift records overlapping a [start_sec, end_sec) window --
        # i.e. everyone who shared any part of this shift with the player.
        return [
            s for s in self._teammate_shifts(player_id, period)
            if time_to_seconds(s["startTime"]) < end_sec and start_sec < time_to_seconds(s["endTime"])
        ]

    def player_actions(self, player_id, max_teammates=5):
        # Every event this player was the actor in (shots, goals, hits, etc.),
        # across the whole game -- not just what happened during their shifts --
        # plus which of their own teammates (skaters only) were on ice for each one.
        # Player1 is always the initiator/owner of the play; Player2..Player(N+1)
        # are the teammates who were on the ice with them for it.
        actions = self.dataframe[self.dataframe["actorId"] == player_id].copy()

        on_ice_names = actions.apply(
            lambda row: self.teammates_on_ice(player_id, row["period"], time_to_seconds(row["timeInPeriod"])),
            axis=1,
        )
        actions["Player1"] = actions["actorName"]
        for i in range(max_teammates):
            actions[f"Player{i + 2}"] = on_ice_names.apply(lambda names, i=i: names[i] if i < len(names) else None)

        return actions

    def line_actions(self, player_id, max_players=6):
        # Every play made by anyone on this player's line (the player himself or
        # a teammate who was on the ice with him), across all of his shifts.
        # Player1 is whoever actually made the play -- could be this player or a
        # teammate; the rest of the line (this player included, if he wasn't the
        # one who made it) fills the remaining slots.
        team_id = self.team_id_of(player_id)
        opponent = self.opponent_abbrev(player_id)
        rows = []
        for shift in self.player_shifts(player_id):
            period = shift["period"]
            start_sec = time_to_seconds(shift["startTime"])
            end_sec = time_to_seconds(shift["endTime"])

            teammates = self.teammates_on_ice_during(player_id, period, start_sec, end_sec)
            line_ids = {t["playerId"] for t in teammates} | {player_id}

            plays = self.plays_during_shift(shift)
            line_plays = plays[plays["actorId"].isin(line_ids)]

            for _, play in line_plays.iterrows():
                time_sec = time_to_seconds(play["timeInPeriod"])
                on_ice_now = [self.players[player_id]] + self.teammates_on_ice(player_id, period, time_sec)
                others = [name for name in on_ice_now if name != play["actorName"]]

                row = play.to_dict()
                row["teamStrength"] = self.team_strength(team_id, play["awaySkaters"], play["homeSkaters"])
                row["opponent"] = opponent
                row["scoreDifferential"] = self.score_differential(team_id, play["awayScore"], play["homeScore"])
                row["elapsedSeconds"] = elapsed_seconds(play["period"], play["timeInPeriod"])
                row["Player1"] = play["actorName"]
                for i in range(max_players - 1):
                    row[f"Player{i + 2}"] = others[i] if i < len(others) else None
                rows.append(row)

        return pd.DataFrame(rows).sort_values("sortOrder").reset_index(drop=True)

    def build_sequences(self, player_id, k=5, feature_cols=None):
        # (context, label) pairs for training: context is the k plays (by this
        # player OR a teammate) immediately before one of this player's own
        # actions; label is what that action turned out to be.
        if feature_cols is None:
            feature_cols = DEFAULT_FEATURE_COLS

        line = self.line_actions(player_id)
        own_idxs = line.index[line["actorId"] == player_id]

        sequences, labels = [], []
        for idx in own_idxs:
            start = max(0, idx - k)
            context = line.loc[start:idx - 1, feature_cols]
            if len(context) < k:
                continue  # not enough prior plays yet (e.g. early in the game)
            sequences.append(context.values.tolist())
            labels.append(line.loc[idx, "typeDescKey"])

        return sequences, labels

    def build_line_actor_sequences(self, player_id, k=5, feature_cols=None):
        # (context, label) pairs for training: context is the k plays immediately
        # before ANY play made by this player's line (hit, shot, takeaway, goal,
        # etc. -- not just goals); label is who made that next play and what it
        # was (always a line member -- opponents' plays never show up here since
        # line_actions is already filtered to this player's line).
        if feature_cols is None:
            feature_cols = DEFAULT_FEATURE_COLS

        line = self.line_actions(player_id)

        sequences, labels = [], []
        for idx in line.index:
            start = max(0, idx - k)
            context = line.loc[start:idx - 1, feature_cols]
            if len(context) < k:
                continue  # not enough prior plays yet (e.g. early in the game)
            sequences.append(context.values.tolist())
            labels.append({
                "actorId": line.loc[idx, "actorId"],
                "actorName": line.loc[idx, "actorName"],
                "typeDescKey": line.loc[idx, "typeDescKey"],
            })

        return sequences, labels

    def build_shot_outcome_sequences(self, player_id, k=5, feature_cols=None):
        # (context, label) pairs for an xG-style model: context is the k prior
        # plays PLUS the shot attempt itself (k+1 timesteps total), so the
        # model sees the shot's own location/angle/distance/situation the way
        # a real expected-goals model would. Label is 1 if that shot was a
        # goal, else 0.
        #
        # Two fields on the shot's own (final) row get corrected so the input
        # can't leak the answer: typeDescKey would literally read "goal" for
        # a scoring shot, so it's masked to a neutral placeholder; and
        # scoreDifferential on a "goal" row already reflects the score AFTER
        # that goal (that's just how the NHL data is captured), so it's
        # replaced with the score as of the previous play instead.
        if feature_cols is None:
            feature_cols = DEFAULT_FEATURE_COLS

        line = self.line_actions(player_id)
        shot_idxs = line.index[line["typeDescKey"].isin(SHOT_EVENT_TYPES)]

        sequences, labels = [], []
        for idx in shot_idxs:
            start = max(0, idx - k)
            if idx - start < k:
                continue  # not enough prior plays yet (e.g. early in the game)

            context = line.loc[start:idx, feature_cols].copy()
            context.loc[idx, "typeDescKey"] = SHOT_OUTCOME_PLACEHOLDER
            if "scoreDifferential" in feature_cols:
                context.loc[idx, "scoreDifferential"] = line.loc[idx - 1, "scoreDifferential"]

            sequences.append(context.values.tolist())
            labels.append(1 if line.loc[idx, "typeDescKey"] == "goal" else 0)

        return sequences, labels

    def line_report(self, player_id, max_teammates=5):
        # For each shift this player took: who was on their line (teammates only,
        # goalie excluded), the player's own plays that shift, and each teammate's
        # own plays that shift ("their main play").
        report = []
        for shift in self.player_shifts(player_id):
            period = shift["period"]
            start_sec = time_to_seconds(shift["startTime"])
            end_sec = time_to_seconds(shift["endTime"])

            teammates = self.teammates_on_ice_during(player_id, period, start_sec, end_sec)
            teammates = teammates[:max_teammates]
            teammate_ids = {t["playerId"] for t in teammates}

            plays = self.plays_during_shift(shift)
            own_plays = plays[plays["actorId"] == player_id]
            teammate_plays = plays[plays["actorId"].isin(teammate_ids)]

            report.append({
                "shift": shift,
                "teammates": [f"{t['firstName']} {t['lastName']}" for t in teammates],
                "own_plays": own_plays,
                "teammate_plays": teammate_plays,
            })
        return report

    def shift_report(self, player_id):
        # For each shift a player took: who else was on the ice, and what happened.
        # Returns a list of {"shift": ..., "on_ice": [...], "plays": DataFrame}.
        report = []
        for shift in self.player_shifts(player_id):
            start_sec = time_to_seconds(shift["startTime"])
            end_sec = time_to_seconds(shift["endTime"])
            on_ice = self.players_on_ice(
                shift["period"], start_sec, end_sec, exclude_player_id=player_id
            )
            plays = self.plays_during_shift(shift)
            report.append({"shift": shift, "on_ice": on_ice, "plays": plays})
        return report

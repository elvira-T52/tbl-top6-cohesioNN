import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from _NHL_game import DEFAULT_FEATURE_COLS

DATA_PATH = os.path.join(os.path.dirname(__file__), "tbl_all_players_sequences.csv")
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "tbl_vocab.json")

K = 5  # sequence length used when the CSV was built (see build_line_actor_sequences)

CATEGORICAL_FEATURES = ["typeDescKey", "zoneCode", "shotType", "teamStrength", "opponent",
                         "Player1", "Player2", "Player3", "Player4", "Player5", "Player6"]
CONTINUOUS_FEATURES = ["xCoord", "yCoord", "shotAngle", "shotDistance", "scoreDifferential", "elapsedSeconds"]
assert set(CATEGORICAL_FEATURES) | set(CONTINUOUS_FEATURES) == set(DEFAULT_FEATURE_COLS)

PAD, UNK = "<PAD>", "<UNK>"

# Which columns share one vocabulary. Player1..Player6 (on-ice teammates) and
# label_actorName (who made the labeled play) are literally the same identity
# space, so they share a vocab -- same for typeDescKey vs label_typeDescKey.
VOCAB_FAMILIES = {
    "typeDescKey": [f"t{t}_typeDescKey" for t in range(K)] + ["label_typeDescKey"],
    "zoneCode": [f"t{t}_zoneCode" for t in range(K)],
    "shotType": [f"t{t}_shotType" for t in range(K)],
    "teamStrength": [f"t{t}_teamStrength" for t in range(K)],
    "opponent": [f"t{t}_opponent" for t in range(K)],
    "playerName": [f"t{t}_Player{p}" for t in range(K) for p in range(1, 7)] + ["label_actorName"],
}

FEATURE_TO_FAMILY = {
    "typeDescKey": "typeDescKey", "zoneCode": "zoneCode", "shotType": "shotType",
    "teamStrength": "teamStrength", "opponent": "opponent",
    **{f"Player{p}": "playerName" for p in range(1, 7)},
}


def build_vocab(series_list):
    # series_list: every column belonging to one family (e.g. all Player slots +
    # label_actorName). Index 0/1 reserved so unseen-at-inference values and
    # genuinely-missing values stay distinguishable after encoding.
    values = pd.concat(series_list).dropna().unique()
    vocab = {PAD: 0, UNK: 1}
    for v in sorted(values):
        vocab[v] = len(vocab)
    return vocab


def encode_series(s, vocab):
    return s.apply(lambda v: vocab[PAD] if pd.isna(v) else vocab.get(v, vocab[UNK]))


def fit_scaler(df, feature_cols):
    # feature_cols: {feature_name: [column at each timestep]}. Pools values
    # across all K timesteps before computing mean/std, since e.g. t0_xCoord
    # and t4_xCoord are the same feature drawn from the same distribution --
    # using just one timestep's stats would throw away 4/5 of the data.
    # mean/std computed from TRAIN rows only; std of 0 (constant column) would
    # divide by zero, so floor it at 1.
    stats = {}
    for feature, cols in feature_cols.items():
        pooled = pd.concat([df[c] for c in cols])
        mean = float(pooled.mean())
        std = float(pooled.std())
        stats[feature] = (mean, std if std > 1e-6 else 1.0)
    return stats


def apply_scaler(series, mean, std):
    # Missing values fill to the train mean, i.e. 0.0 after scaling -- a
    # neutral value rather than an arbitrary raw-unit default like 0.
    return (series.fillna(mean) - mean) / std


def split_by_game(df, seed=42, train_frac=0.7, val_frac=0.15):
    # Splitting by game (not by row) keeps overlapping context windows from the
    # same game entirely on one side of the split -- shuffling rows directly
    # would leak information between train and test.
    game_ids = df["game_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)
    n_train = int(len(game_ids) * train_frac)
    n_val = int(len(game_ids) * val_frac)
    train_ids = set(game_ids[:n_train])
    val_ids = set(game_ids[n_train:n_train + n_val])
    test_ids = set(game_ids[n_train + n_val:])
    return (
        df[df["game_id"].isin(train_ids)].reset_index(drop=True),
        df[df["game_id"].isin(val_ids)].reset_index(drop=True),
        df[df["game_id"].isin(test_ids)].reset_index(drop=True),
    )


class HagelSequenceDataset(Dataset):
    # One sample = one (k-play context -> next play) example. __getitem__
    # returns a dict of tensors so torch's default_collate stacks each key
    # into a batch automatically.
    def __init__(self, categorical, continuous, label_actor, label_type):
        self.categorical = categorical  # {feature_name: (N, K) long}
        self.continuous = continuous    # (N, K, len(CONTINUOUS_FEATURES)) float
        self.label_actor = label_actor  # (N,) long
        self.label_type = label_type    # (N,) long

    def __len__(self):
        return len(self.label_actor)

    def __getitem__(self, idx):
        return {
            **{name: t[idx] for name, t in self.categorical.items()},
            "continuous": self.continuous[idx],
            "label_actor": self.label_actor[idx],
            "label_type": self.label_type[idx],
        }


def _encode_split(df, vocabs, scaler):
    categorical = {}
    for feature in CATEGORICAL_FEATURES:
        family = FEATURE_TO_FAMILY[feature]
        cols = [f"t{t}_{feature}" for t in range(K)]
        encoded = np.stack(
            [encode_series(df[c], vocabs[family]).to_numpy() for c in cols], axis=1
        )
        categorical[feature] = torch.tensor(encoded, dtype=torch.long)

    continuous = np.stack([
        np.stack([
            apply_scaler(df[f"t{t}_{feature}"], *scaler[feature]).to_numpy()
            for feature in CONTINUOUS_FEATURES
        ], axis=1)
        for t in range(K)
    ], axis=1)
    continuous = torch.tensor(continuous, dtype=torch.float32)

    label_actor = torch.tensor(
        encode_series(df["label_actorName"], vocabs["playerName"]).to_numpy(), dtype=torch.long
    )
    label_type = torch.tensor(
        encode_series(df["label_typeDescKey"], vocabs["typeDescKey"]).to_numpy(), dtype=torch.long
    )

    return HagelSequenceDataset(categorical, continuous, label_actor, label_type)


def prepare_datasets(data_path=DATA_PATH, vocab_path=VOCAB_PATH, seed=42):
    df = pd.read_csv(data_path)
    train_df, val_df, test_df = split_by_game(df, seed=seed)

    vocabs = {
        family: build_vocab([train_df[c] for c in cols])
        for family, cols in VOCAB_FAMILIES.items()
    }
    scaler = fit_scaler(
        train_df, {feature: [f"t{t}_{feature}" for t in range(K)] for feature in CONTINUOUS_FEATURES}
    )

    with open(vocab_path, "w") as f:
        json.dump({"vocabs": vocabs, "scaler": scaler}, f, indent=2)

    return {
        "train": _encode_split(train_df, vocabs, scaler),
        "val": _encode_split(val_df, vocabs, scaler),
        "test": _encode_split(test_df, vocabs, scaler),
        "vocabs": vocabs,
        "scaler": scaler,
    }


if __name__ == "__main__":
    datasets = prepare_datasets()

    print("vocab sizes:")
    for family, vocab in datasets["vocabs"].items():
        print(f"  {family}: {len(vocab)}")

    for split in ["train", "val", "test"]:
        ds = datasets[split]
        print(f"\n{split}: {len(ds)} examples")

    sample = datasets["train"][0]
    print("\nfirst training sample shapes:")
    for k, v in sample.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

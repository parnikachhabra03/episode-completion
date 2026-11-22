"""Simulate streaming sessions, then damage them the way real logs get damaged.

There's no public BBC viewing log, so this generates one. The point of the
project is the pipeline (validate, clean, model, check fairness), which works
the same on any session-level table. The "true" pattern is written down here
so you can check whether the model found it.
"""
import numpy as np
import pandas as pd

GENRES = ["Drama", "Comedy", "Documentary", "Entertainment", "Children", "News", "Sport"]
DEVICES = ["TV", "Mobile", "Tablet", "Desktop"]
AGE_BANDS = ["16-24", "25-34", "35-54", "55+"]


def simulate(n: int = 20_000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "session_id": np.arange(n),
        "user_id": rng.integers(0, 4_000, n),
        "age_band": rng.choice(AGE_BANDS, n, p=[0.2, 0.25, 0.3, 0.25]),
        "device": rng.choice(DEVICES, n, p=[0.45, 0.3, 0.15, 0.1]),
        "genre": rng.choice(GENRES, n),
        "episode_minutes": rng.choice([15, 30, 45, 60, 90], n, p=[0.15, 0.35, 0.2, 0.25, 0.05]),
        "start_hour": rng.integers(0, 24, n),
        "episode_number": rng.integers(1, 11, n),
        "watched_previous": rng.random(n) < 0.55,
        "is_weekend": rng.random(n) < 2 / 7,
    })
    # The hidden rules the model should recover.
    logit = (
        0.9
        + 0.8 * (df["device"] == "TV")
        - 0.6 * (df["device"] == "Mobile")
        - 0.025 * (df["episode_minutes"] - 30)
        + 1.1 * df["watched_previous"]
        + 0.5 * df["start_hour"].between(19, 22)
        - 0.7 * df["start_hour"].between(0, 5)
        + 0.3 * df["is_weekend"]
        + df["genre"].map({"Drama": 0.4, "Children": 0.5, "News": -0.5}).fillna(0)
        - 0.08 * (df["episode_number"] - 1)
    )
    df["completed"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    frac = np.where(df["completed"], rng.uniform(0.9, 1.0, n), rng.beta(2, 3, n) * 0.9)
    df["minutes_watched"] = (frac * df["episode_minutes"]).round(1)
    start = pd.Timestamp("2026-09-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D")
    df["started_at"] = start + pd.to_timedelta(df["start_hour"], unit="h")
    return df


def add_mess(df: pd.DataFrame, seed: int = 3) -> pd.DataFrame:
    """Inject the faults real logs have, so the cleaning step has work to do."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    df["device"] = df["device"].astype(object)
    pick = lambda frac: rng.choice(n, int(n * frac), replace=False)  # noqa: E731
    df.loc[pick(0.03), "device"] = None                            # missing
    df.loc[pick(0.04), "device"] = df["device"].str.lower()        # inconsistent case
    df.loc[pick(0.02), "device"] = "Smart TV"                      # synonym
    df.loc[pick(0.01), "minutes_watched"] = -1                     # impossible
    over = pick(0.01)
    df.loc[over, "minutes_watched"] = df.loc[over, "episode_minutes"] * 3  # longer than episode
    df.loc[pick(0.02), "age_band"] = "unknown"
    dupes = df.sample(frac=0.015, random_state=seed)               # double-logged
    return pd.concat([df, dupes]).sample(frac=1, random_state=seed).reset_index(drop=True)

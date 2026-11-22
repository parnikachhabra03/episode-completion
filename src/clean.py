"""Validate and clean session data, logging every decision.

Each rule records how many rows it touched, so the cleaning is auditable
instead of a silent `dropna()`.
"""
import pandas as pd

DEVICE_MAP = {"tv": "TV", "smart tv": "TV", "mobile": "Mobile",
              "tablet": "Tablet", "desktop": "Desktop"}


def clean(df: pd.DataFrame):
    log = []
    out = df.copy()

    def note(rule, rows, action):
        log.append({"rule": rule, "rows": int(rows), "action": action})

    before = len(out)
    out = out.drop_duplicates(subset="session_id")
    note("Duplicate session_id", before - len(out), "kept first copy")

    raw = out["device"]
    mapped = raw.str.lower().map(DEVICE_MAP)
    note("Device spelled differently", int((raw.notna() & (mapped != raw)).sum()),
         "standardised to TV / Mobile / Tablet / Desktop")
    note("Device missing", int(raw.isna().sum()), "labelled 'Unknown' (kept as its own group)")
    out["device"] = mapped.fillna("Unknown")

    bad = (out["minutes_watched"] < 0) | (out["minutes_watched"] > out["episode_minutes"] * 1.05)
    note("Minutes watched impossible (<0 or > episode length)", int(bad.sum()),
         "dropped: the target can't be trusted")
    out = out[~bad]

    unknown_age = out["age_band"].eq("unknown")
    note("Age band unknown", int(unknown_age.sum()),
         "kept for training, excluded from the fairness table")

    return out.reset_index(drop=True), pd.DataFrame(log)


def validate(df: pd.DataFrame) -> list[str]:
    """Checks that must pass after cleaning. Returns a list of failures."""
    problems = []
    if df["session_id"].duplicated().any():
        problems.append("duplicate session_id")
    if not set(df["device"]).issubset({"TV", "Mobile", "Tablet", "Desktop", "Unknown"}):
        problems.append("unexpected device value")
    if (df["minutes_watched"] < 0).any():
        problems.append("negative minutes")
    if (df["minutes_watched"] > df["episode_minutes"] * 1.05).any():
        problems.append("minutes longer than episode")
    if df[["episode_minutes", "start_hour", "completed"]].isna().any().any():
        problems.append("missing values in required columns")
    return problems

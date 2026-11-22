"""Predict whether a viewer finishes an episode, from what's known at the
start of the session (no peeking at minutes_watched, which would leak the
answer)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["device", "genre", "age_band"]
NUMERIC = ["episode_minutes", "start_hour", "episode_number"]
BINARY = ["watched_previous", "is_weekend"]
FEATURES = CATEGORICAL + NUMERIC + BINARY
LEAKY = ["minutes_watched"]  # recorded after the outcome: never a feature


def add_features(df):
    df = df.copy()
    df["prime_time"] = df["start_hour"].between(19, 22)
    df["late_night"] = df["start_hour"].between(0, 5)
    return df


def build(kind: str):
    feats = NUMERIC + BINARY + ["prime_time", "late_night"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), feats),
    ])
    if kind == "logistic":
        clf = LogisticRegression(max_iter=1000)
    elif kind == "boosting":
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


def split(df, seed=0):
    """Split by viewer so the same person never appears in train and test."""
    users = df["user_id"].unique()
    train_u, test_u = train_test_split(users, test_size=0.25, random_state=seed)
    return df[df["user_id"].isin(train_u)], df[df["user_id"].isin(test_u)]


def fit_and_score(train, test):
    train, test = add_features(train), add_features(test)
    y_tr, y_te = train["completed"].astype(int), test["completed"].astype(int)
    results, models = [], {}
    base = np.full(len(y_te), y_tr.mean())
    results.append({"model": "Baseline (overall rate)", "roc_auc": 0.5,
                    "brier": brier_score_loss(y_te, base)})
    for kind in ["logistic", "boosting"]:
        m = build(kind).fit(train, y_tr)
        p = m.predict_proba(test)[:, 1]
        results.append({"model": kind, "roc_auc": roc_auc_score(y_te, p),
                        "brier": brier_score_loss(y_te, p)})
        models[kind] = m
    return pd.DataFrame(results), models, test


def fairness_table(model, test, threshold=0.5):
    """Does the model work equally well for every age group?"""
    test = test[test["age_band"] != "unknown"]
    p = model.predict_proba(test)[:, 1]
    rows = []
    for band, g in test.assign(p=p).groupby("age_band"):
        y = g["completed"].astype(int)
        rows.append({"age_band": band, "sessions": len(g),
                     "actual_rate": y.mean(), "predicted_rate": (g["p"] >= threshold).mean(),
                     "roc_auc": roc_auc_score(y, g["p"])})
    t = pd.DataFrame(rows)
    t["auc_gap_vs_best"] = t["roc_auc"].max() - t["roc_auc"]
    return t


def importance(model, test, n_repeats=5):
    y = test["completed"].astype(int)
    r = permutation_importance(model, test, y, scoring="roc_auc",
                               n_repeats=n_repeats, random_state=0)
    cols = list(test.columns)
    imp = pd.Series(r.importances_mean, index=cols)
    keep = FEATURES + ["prime_time", "late_night"]
    return imp[keep].sort_values(ascending=False)

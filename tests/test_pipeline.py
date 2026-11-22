import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from clean import clean, validate  # noqa: E402
from model import FEATURES, LEAKY, fit_and_score, split  # noqa: E402
from simulate import add_mess, simulate  # noqa: E402


@pytest.fixture(scope="module")
def data():
    raw = add_mess(simulate(n=6000, seed=5), seed=5)
    df, log = clean(raw)
    return raw, df, log


def test_mess_is_detected_before_cleaning(data):
    raw, _, _ = data
    assert validate(raw), "the dirty data should fail validation"


def test_clean_data_passes_validation(data):
    _, df, _ = data
    assert validate(df) == []


def test_cleaning_log_counts_every_rule(data):
    raw, df, log = data
    dropped = log.loc[log.rule.str.startswith(("Duplicate", "Minutes")), "rows"].sum()
    assert len(raw) - len(df) == dropped
    assert (log["rows"] > 0).all()


def test_no_leaky_feature():
    assert not set(LEAKY) & set(FEATURES)


def test_viewer_split_has_no_overlap(data):
    _, df, _ = data
    tr, te = split(df)
    assert not set(tr.user_id) & set(te.user_id)


def test_models_beat_baseline(data):
    _, df, _ = data
    scores, _, _ = fit_and_score(*split(df))
    base = scores.iloc[0]
    for _, r in scores.iloc[1:].iterrows():
        assert r["roc_auc"] > 0.65
        assert r["brier"] < base["brier"]

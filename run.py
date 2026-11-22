"""Full pipeline: simulate -> mess up -> clean -> validate -> model -> fairness
-> outputs/model_card.md and outputs/importance.png.

    python run.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from clean import clean, validate  # noqa: E402
from model import fairness_table, fit_and_score, importance, split  # noqa: E402
from simulate import add_mess, simulate  # noqa: E402

OUT = Path(__file__).parent / "outputs"


def main():
    raw = add_mess(simulate())
    df, log = clean(raw)
    problems = validate(df)
    assert not problems, problems
    print(f"Raw rows: {len(raw):,} -> clean rows: {len(df):,}")
    print(log.to_string(index=False))

    train, test = split(df)
    scores, models, test_f = fit_and_score(train, test)
    best = scores.iloc[1:].sort_values("roc_auc", ascending=False).iloc[0]["model"]
    fair = fairness_table(models[best], test_f)
    imp = importance(models[best], test_f)
    print(scores.round(3).to_string(index=False))
    print(fair.round(3).to_string(index=False))
    print(imp.round(4).to_string())

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.iloc[::-1].plot.barh(ax=ax, color="#1f5fa8")
    ax.set_xlabel("Drop in ROC AUC when the feature is shuffled")
    ax.set_title("What predicts finishing an episode")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "importance.png", dpi=150)

    (OUT / "model_card.md").write_text(f"""# Model card: episode completion

**Question.** At the start of a session, how likely is this viewer to finish the episode?
**Data.** {len(raw):,} simulated sessions, {len(df):,} after cleaning. Split by viewer (75/25), so no one is in both sets.
**Not used as a feature.** `minutes_watched`, because it's recorded after the outcome and would leak the answer.

## Cleaning log
{log.to_markdown(index=False)}

## Results (test set)
{scores.round(3).to_markdown(index=False)}

Chosen model: **{best}**.

## Does it work equally well for every age group?
{fair.round(3).to_markdown(index=False)}

## What drives it
![importance](importance.png)

## Limits and responsible use
- Simulated data: the patterns are planted, so this shows the method, not real BBC audience behaviour.
- Use it to improve the product (e.g. better "continue watching" or episode-length choices), not to target or exclude groups of viewers.
- Re-check the age-group table on every retrain; a gap above 0.05 AUC should block release.
""")
    print(f"Wrote {OUT/'model_card.md'}")


if __name__ == "__main__":
    main()

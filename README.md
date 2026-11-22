# Will They Finish the Episode? Clean, model and fairness-check viewing data

This project runs a full data science lifecycle on 20,000 simulated streaming
sessions: validate, clean, build features, model, evaluate, then check whether
the model works equally well for every age group. It ends with a model card.

The data is simulated because no public BBC viewing log exists. Faults are
deliberately injected: duplicates, missing and misspelt devices, impossible
watch times and unknown ages. The cleaning step has to catch and log each one.

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib tabulate pytest
python run.py      # writes outputs/model_card.md and outputs/importance.png
pytest -q          # 6 tests
```

## What happens

1. **Simulate** sessions with hidden rules. For example, TV viewers and people
   who watched the previous episode finish more often. Long episodes and late
   nights lower completion.
2. **Mess them up** (`add_mess`), then **clean** them with a logged rule for
   each fault. From the last run:

| Rule | Rows | Action |
| --- | --- | --- |
| Duplicate session_id | 300 | kept first copy |
| Device spelled differently | 1,157 | standardised |
| Device missing | 584 | labelled "Unknown" |
| Minutes watched impossible | 399 | dropped |
| Age band unknown | 392 | excluded from fairness table only |

3. **Validate:** five checks must pass after cleaning. The tests confirm they
   fail on the raw data.
4. **Model:** logistic regression vs gradient boosting, with a viewer-level
   split so no one appears in both train and test. `minutes_watched` is
   excluded because it would leak the answer.

| Model | ROC AUC | Brier (lower is better) |
| --- | --- | --- |
| Baseline | 0.500 | 0.205 |
| Logistic regression | 0.756 | 0.169 |
| Gradient boosting | 0.741 | 0.173 |

   The simple model wins because the planted pattern is close to linear.
   Knowing when not to reach for the complex model is the point.
5. **Fairness:** AUC per age band ranges from 0.750 to 0.765. The largest gap
   is 0.015, under the 0.05 release limit set in the model card.
6. **Explain:** permutation importance ranks device and "watched previous
   episode" highest. Those are two of the rules planted in the simulation, so
   the model found what was really there.

## Make it yours (do at least two before you list it)

1. **Find the planted rule it missed.** Compare `importance.png` with the
   logit in `simulate.py`. Is anything planted but ranked low? Why?
2. **Threshold choice.** Predicted completion (about 85%) is higher than actual
   (about 71%) at a 0.5 threshold. Pick a threshold that matches the real rate
   and explain the trade-off.
3. **Calibration plot.** Use `sklearn.calibration.CalibrationDisplay`. Is the
   logistic model's "70%" really 70%?
4. **A product recommendation.** Write 5 lines for a product manager: what
   should iPlayer change, based on the drivers?
5. **Swap in real data.** Any session-level table with an outcome works, such
   as a Kaggle streaming or churn dataset. Rewrite `clean.py` rules to fit it.

## Things to be ready to explain

- Data leakage, and why `minutes_watched` can't be a feature.
- Why the split is by viewer, not by row.
- ROC AUC vs Brier score: ranking vs probability quality.
- Why "Unknown" devices are kept as a group instead of dropped.

# Model card

## Intended use

Educational portfolio demonstration of DE/MLE for historical credit-risk data. Scores are model outputs; they are not approved lending decisions or established calibrated probabilities for a live population.

## Models and selection

Logistic regression is the baseline. LightGBM uses 120 trees, learning rate 0.04, maximum depth 5, 15 leaves, minimum child size 30, seed 42 and two training threads. Numerical features use training-only median imputation and standardization; categoricals use training-only mode imputation and ordinal encoding with an unknown value.

Three expanding forward-validation folds each use four validation weeks. Candidates are ranked by mean fold stability only when all fold stability metrics are defined. Full and internal modes are selected independently. A negative stability value is a legitimate result and is not hidden. The final four observed weeks were used and published by the original project; subsequent analyses describe them as a retrospective, previously inspected evaluation period, not a newly untouched independent test.

For each selected base model, strictly out-of-time development predictions are divided into a calibration-fit window and a later method-selection window. Raw, Platt/sigmoid and isotonic results are compared by Brier score with log loss as a tie-breaker; the selected method is then refitted on all development out-of-time predictions. The final evaluation labels are never used to fit or select a calibrator. Full and internal calibrators have distinct versions bound to their base-model version. `score` remains the legacy raw probability; new consumers should use the explicit `raw_score` and `calibrated_score` fields.

The [competition metric](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview) combines mean weekly Gini, an 88× penalty on negative trend, and a 0.5× population-standard-deviation residual penalty. The implementation follows sorted observed-week ordinal positions. It reports unavailable stability when a week has fewer than 20 observations, only one target class, or fewer than two evaluated weeks. This explicit local validity policy supplements the competition formula.

## Validation and monitoring

Reported metrics include overall and weekly AUC/Gini, Brier, log loss, ECE, O/E, calibration intercept/slope, trend slope, residual variation and stability. Calibration bins use development-frozen score quantiles and include application/event counts. Segment tables freeze development boundaries and mark low-count slices not estimable; they are robustness checks, not fairness certification. Feature and score PSI above 0.2 is a demonstration alert, not proof of failure or an automatic retraining trigger. External bureau failure routes explicitly to a separately trained internal model.

## Limitations

- Label arrival and source availability are simulated; temporal splits alone cannot prove point-in-time correctness.
- No customer-disjoint split is claimed because case_id identifies applications.
- The model uses a curated feature subset and no exhaustive search or leaderboard optimization.
- Source absence can be correlated with risk; proxy discrimination and subgroup performance require further analysis before any real-world use.
- Activation requires computable metrics and a compatible bundle, not a business-approved risk threshold.
- Model artifacts are pickle-compatible trusted files; do not load artifacts from unknown parties.
- API deadlines stop waiting for a response; Python worker-thread computation can continue after timeout. Production hard cancellation needs process isolation.

See [measured results](../reports/RESULTS.md) for dataset scope and observed performance. Synthetic demonstration scores must not be interpreted as competition scores.

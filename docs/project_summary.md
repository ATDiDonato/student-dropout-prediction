# Student Dropout Prediction — Project Summary

## Objective

The project tests how dropout classification changes as additional information becomes available: intake data, attendance data and assessment data. It compares XGBoost with a multilayer-perceptron neural network and gives recall additional weight because a missed at-risk learner may represent a missed opportunity for intervention.

## Data audit

- 25,059 enrolment records and 24,877 learner identifiers.
- 3,754 dropout records (14.98%).
- 182 learners have two enrolment records.
- 36 repeated learners have different outcomes across their enrolments.
- No exact duplicate rows.
- `CompletedCourse` is exactly the inverse of `dropout` and is removed from every feature set.

The repeated learners make a standard random row split unsafe. One learner could otherwise appear in both training and test data, allowing the model to learn learner-specific patterns that would not generalise.

## Method

One stratified, learner-grouped split is created and reused for every experiment:

| Split | Rows | Learners | Dropouts | Dropout rate |
|---|---:|---:|---:|---:|
| Train | 15,035 | 14,925 | 2,252 | 14.98% |
| Validation | 5,012 | 4,976 | 751 | 14.98% |
| Test | 5,012 | 4,976 | 751 | 14.98% |

Preprocessing is fitted on training data only. Categorical variables are imputed and one-hot encoded; infrequent values are grouped. Numeric values are median-imputed and standardised. XGBoost uses a training-derived positive-class weight. The neural network uses balanced sample weights and early stopping.

The classification threshold is selected on validation data by maximising the F2 score. This weights recall twice as strongly as precision. The held-out test set is not used for preprocessing, threshold selection or model fitting.

## Results

### Full eligible population at each stage

| Stage | Model | ROC-AUC | Average precision | Precision | Recall | Specificity | F2 |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage 1 | XGBoost | 0.893 | 0.686 | 0.551 | 0.790 | 0.887 | 0.727 |
| Stage 1 | Neural network | 0.890 | 0.674 | 0.580 | 0.750 | 0.904 | 0.708 |
| Stage 2 | XGBoost | 0.930 | 0.771 | 0.602 | 0.828 | 0.903 | 0.770 |
| Stage 2 | Neural network | 0.920 | 0.724 | 0.547 | 0.822 | 0.880 | 0.747 |
| Stage 3 | XGBoost | 0.998 | 0.977 | 0.765 | 0.976 | 0.980 | 0.925 |
| Stage 3 | Neural network | 0.996 | 0.955 | 0.647 | 0.976 | 0.964 | 0.886 |

Stage 2 adds useful predictive signal. On the complete test population, the XGBoost model improves ROC-AUC from 0.893 to 0.930 and average precision from 0.686 to 0.771, while recall increases from 0.790 to 0.828.

### Stage 3 eligibility and leakage control

All 2,231 records without assessment data are dropouts. Missing assessment records therefore reveal an outcome-adjacent condition rather than a safe predictive feature. These records are excluded from the Stage 3 experiment.

Stage 3 consequently covers 22,828 of 25,059 records (91.1%) and excludes 2,231 of 3,754 dropouts (59.4%). Its test set contains 4,552 records and 291 dropouts. Passed and failed module counts dominate the model importance, so the excellent Stage 3 discrimination should be interpreted as late-stage risk classification among learners with assessments—not early-warning prediction.

For a like-for-like comparison on the Stage 3-eligible test cohort, XGBoost ROC-AUC rises from 0.854 at Stage 1 to 0.902 at Stage 2 and 0.998 at Stage 3. Average precision rises from 0.419 to 0.551 and 0.977 respectively.

## Conclusions

1. Intake information produces a useful but imperfect early model.
2. Attendance adds meaningful signal and improves both ranking performance and recall-weighted classification.
3. Assessment outcomes are extremely predictive, but arrive late and do not cover many learners who have already dropped out.
4. XGBoost outperforms the neural network at every stage in this tabular-data setting.
5. A practical system would need explicit prediction dates, feature-availability rules, probability calibration, fairness review and prospective validation before intervention decisions.

## Limitations

- No event dates are supplied, so temporal availability cannot be verified directly.
- Stage 3 evaluates a narrower population and is not directly comparable with all-record Stage 1/2 results without the separate common-cohort analysis.
- Progression destination fields are treated as intended pathways known at intake, following the project documentation; this assumption requires confirmation from the data owner.
- Fixed model configurations were used to emphasise reproducibility and stage comparison. They are not claimed to be globally optimal.
- No causal claim is made: model importance describes predictive association, not reasons for dropout.
- The analysis does not establish that interventions based on predictions would improve learner outcomes.


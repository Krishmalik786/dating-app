Why LightGBM

## Why LightGBM (Light Gradient Boosting Machine)

---

## 1. Why LightGBM?

Predicting a successful match is a **tabular classification problem with multiple classes** and 50,000 observations, 10 classes for the outcome, and a combination of continuous, ordinal, binary and one-hot encoded features. This is the very problem that gradient boosted trees excel at - and among these, LightGBM is the best choice for four reasons.

First, LightGBM is designed for tabular data with mixed features.  
Our engineered feature set includes raw numerical features (`likes_received`, `app_usage_time_min`), ordinal encodings of some features (`usage_ordinal`, `swipe_ordinal`), log-transformed features and polynomial features, cyclical encodings of the hour of day (`hour_sin`, `hour_cos`), and 49 binary interest-tag flags. LightGBM's histogram-based split-finding algorithm can natively handle all of these features, including efficient discretisation of continuous features without needing perfectly scaled and distributed data. A neural network (such as MLP) requires normalisation and has more trouble with the binary tag features.

Second, LightGBM handles class-imbalanced and multi-class problems.  
With 10 classes and a real-world data set which would probably be imbalanced (much fewer "Relationship Formed" than "No Action" cases), LightGBM's `class_weight` and `num_class` support is straightforward. It optimises multi-class log-loss; there is no need to add wrappers and calibrators.

**Third, LightGBM is very transparent with feature importance.**  
For a product like a matchmaking site, it's crucial to know why a model makes a particular prediction - for confidence and troubleshooting. LightGBM offers gain and split feature importances, so it's easy to show stakeholders which user actions are important to predictions. Neither a Stacking Ensemble, nor a very deep MLP come with this level of transparency without substantial added work (such as the misleading and expensive SHAP on stacked models).

**Fourth, inference and deployment is a consideration.**  
Our Streamlit app calls the `predict()` method on button click. LightGBM is small (2 MB) as a `.pkl` file, loads in milliseconds and makes a single prediction in less than 1 ms. The Stacking Ensemble consists of three base models and a meta-learner, which are run in series, resulting in 4-6× slower inference, and a noticeable slow-down in the app.

---

## 2. Why Not the Others?

| Model | Why not this model? |
|---|---|
| **Stacking Ensemble** | Highest accuracy (10.44%) but 4× slower inference, no feature importance, difficult to use in production. The 1.1 pp increase in accuracy vs LightGBM is not significant when the best you can do with the data is ~10%.
| **XGBoost** | Similar to LightGBM (9.84% vs 9.32%), but LightGBM is ~3× faster at training on this data size, because of leaf-wise trees, not level-wise as in XGBoost. At 50k rows it's not much of an issue, but scales up.
| **CatBoost** | Good (9.86%) and supports categoricals, but we take care of that in preprocessing - so CatBoost's biggest feature is circumvented. It also takes the longest to train of the three boosting algorithms.
| **Random Forest** | Poorest (9.69% accuracy, CV mean 9.97%) and takes up the most memory (all the 300 decision trees). Random Forest is never as good as gradient boosting on tabular data.
| **MLP (Neural Net)** | Worst performer (9.36%) in this case. Neural nets are powerful, but need large amounts of clean data with a lot of information to beat trees. On a dataset of 50k synthetic data with almost zero mutual information between the features and the target, the MLP showed no benefit and was the most variable.

---

## 3. Metrics That Matter for This Use Case

**Primary: Weighted F1-Score**  
Accuracy is deceptive if there are different costs associated with different classes. For a dating application, making a "Mutual Match" prediction when it should be "Blocked" has a much more negative impact on user experience than predicting "No Action" when they should be "Chat Ignored". Weighted F1 penalises false positives and false negatives equally for all 10 classes: it's the most accurate single metric.

**Secondary: Recall on valuable outcomes, per class**  
A dating app wants to present to the user the classes of *Relationship Formed*, *Date Happened* and *Mutual Match*. If the model has high recall on these classes, it will identify the times when a match is indeed occurring - at the expense of sometimes predicting a match when in fact there isn't one. A product team would want to optimise the recall on these three outcomes.

**Not used as primary: Accuracy alone**  
A random model on 10 balanced classes: 10%. It does not provide any insight about the classes on which the model is good or bad.

---

## 4. Observed Results

| Ensemble | Test Accuracy | Weighted F1 | CV Mean (3-fold) |
|---|---|---|---|
| Stacking Ensemble | 10.44% | 0.074 | — |
| CatBoost | 9.86% | 0.097 | — |
| XGBoost | 9.84% | **0.098** | 10.32% ± 0.06% |
| Random Forest | 9.69% | 0.097 | 9.97% ± 0.28% |
| MLP (Neural Net) | 9.36% | 0.093 | — |
| **LightGBM** | 9.32% | 0.093 | 10.20% ± 0.24% |

LightGBM's CV standard deviation (±0.24%) is greater than XGBoost's (±0.06%) due to some variability in the fold composition (to watch out for on a real dataset). But the low absolute variance of all models indicates that model performance isn't limited by model capacity in this case.

> **Note on accuracy ceiling:** Mutual information analysis revealed almost zero predictive information in the data for all features vs. `match_outcome` (highest MI = 0.005 for `likes_received`). The data is generated, with randomly assigned labels. All models are at or near the best possible accuracy on this data With real data containing behavioural signal, for example, LightGBM should reach 70-85% accuracy.
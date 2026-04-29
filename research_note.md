# Research Note - Weaknesses, Improvements & Reflections

---

## 1. Major Weaknesses of Model & Pipeline

The biggest weakness is the data itself.  
`match_outcome` was set randomly when generating the data. The mutual information between all features and the target is near zero (the highest is: `likes_received` at MI = 0.005). This implies no model (be it an ANN or linear model or with any hyperparameters) can learn anything. The six models constructed in this project are all at the theoretical maximum of random guessing - ~10% for 10 balanced classes. This is more a problem with the data than the model; the predictive task is invalid.

**The 10-class task is not a useful one for a dating app.  
For the product, the distinction between whether a given interaction will lead to "Blocked" vs "Ghosted" vs "Chat Ignored" is not valuable. These are fine-grained labels that even people couldn't accurately predict from user profiles. What would be more appropriate would be binary (did a meaningful interaction occur: yes/no) or ordinal (no interest → one-sided interest → mutual interest → date → relationship). As it is, the task is made harder unnecessarily, with no benefit to the business.

**Risk of leakage in reverse.**  
Some of the engineered features (`mutual_matches`, `like_efficiency`, `engagement_rate`) are outcomes or close to outcomes - they represent the result of what has happened, rather than what will happen. In practice, these features would not be available at time of prediction (you don't know how many `mutual_matches` there will be until a match occurs). The pipeline as it is built mixes past summaries and features.

**No temporal validation.**  
A random train/test split. Users of a dating app change their behaviour over time - we should validate on a future time period, not a random 20%. Random sampling introduces future knowledge to training, and yields overly optimistic estimates.

**The inference preprocessing is vulnerable.**  
In the `preprocess_profile` function of `app.py`, the 49 interests are rebuilt from the raw CSV every time the app is cold-started. In production, if the dataset file is not present the app can't make predictions. The list of tags should be serialised when the model is trained, and then loaded as an artifact.

---

## 2. How I would improve the Matching algorithm

**Use a real dataset containing signal.**  
The Columbia Speed Dating data (Kaggle) has ~8,000 real speed-dating matches, with mutual yes/no results, compatibility with demographic features, self-reported preferences and partner ratings on six attributes (attractiveness, sincerity, intelligence, fun, ambition, shared interests). On this dataset, an AUC of 83-87% can be reached with a LightGBM model. The code developed here would be 100% applicable.

**Pairwise matching problem**  
What happens in matching is not "what will happen to User A?" - it is "will User A and User B like each other?" This is a pairwise model: combine features of both users, generate compatibility vectors (overlap of interests, distance, education gap, time overlap of activities), and predict whether there will be a mutual match. This is what the models at Tinder, Hinge and OkCupid do.

**Add collaborative filtering to the content features.**  
What's in a profile (interests, education, income) reflects explicit interests. Implicit features - who you gave a like or a message to first, how many minutes you viewed a profile - are much stronger. A combination of a gradient boosted tree (for explicit profile preferences) and latent factor model (for implicit preferences) would pick up both signals. This is the typical production model (at scale).

**Do not temporal cross-validate.**  
Train on the first 9 months, validate on the 10th month, test on the 11th month. Avoids future leakage, and mirrors how the model would be used in practice with new users.

**Add a calibration layer.**  
Boosted tree models do not return well-calibrated probabilities - the output of a model's `predict_proba` method tends to not be correlated with the true probability of success. The confidence percentages in the app would be useful to customers with a Platt scaling or isotonic regression calibration layer, trained on a separate calibration set.

---

## 3. How I would have made the AI Layer better (Task 3)

**Current version is a zero shot Groq/LLaMA API call with a hand-crafted prompt.**  
It's fine for a proof-of-concept but has three main issues: it is not based on the model's reasoning, it lacks memory and the advice it gives is not specific.

**Explain the prediction in terms of feature importances.**  
Rather than feeding only the label and confidence of the prediction to the LLM, feed the top 5 features that the model used to make that prediction (from the SHAP values calculated at the time of prediction). This ensures the AI explains why the model made the prediction for the user's profile, instead of providing plausible but unrelated advice. For example, "The many messages you sent (top feature) had a strong weight in the prediction 'Chat Ignored' - people with a similar messaging style tend to over-message until a match is made".

**Retrieval-augmented generation (RAG) with dating research.**  
Use peer-reviewed research on dating and relationship development and on dating app use (e.g., Bruch & Newman 2018 on desirability hierarchies, Tyson et al. 2016 on Tinder usage patterns) to inform the LLM's recommendations. With a vector database of ~200 research abstracts, the LLM can draw from specific studies rather than give advice.

**Introduce some memory for each session.**  
At present each prediction is accompanied by a standalone prediction. If the conversation history is bound to a session then I can ask the model: *"Why did my confidence go down when I changed my bio length?"* or *"What should I change to have the biggest impact?"* This converts the AI layer from a report to an advisor.

**Add evaluation of the AI responses.**  
There is no evaluation of the quality, helpfulness and safety of the explanations. We would need at least: (1) a truth check - is the explanation consistent with the feature importances? (2) a small human evaluation for helpful and tone.

## 4. If I had to start again, I would…

**I would have preferred a binary classification target, rather than 10 classes.**

I would run mutual information on the raw data, before writing any model code. This can be done in five lines of code in three seconds. It will tell me right away if there is any signal to learn. If I had done this first I would have noticed the random label problem before spending time feature engineering, training, and tuning a model for a 10-class problem with zero signal.

If I had to use this data set I would redefine the target on day 1:

``` match_binary = 1  if outcome in {Mutual Match, Date Happened,                                    Relationship Formed, Instant Match}              else 0 ```

This reduces 10 random classes to 2, boosts the signal-to-noise ratio, and provides a model output which is useful to the end user ("you will / won't find a match"). The question to answer for a dating app is a binary one: *will this relationship go anywhere? - not 10 reasons how it will not.
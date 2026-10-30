# 🌲 Ensemble Learning in Practice: Bagging & Boosting on Heart Disease Data

> One model can be wrong. A crowd of models, combined well, is usually wrong less often.

In this post I walk through **ensemble learning** using a real example: predicting heart disease from patient data (918 patients, 12 features). Everything here comes from my own notebook, so you can reproduce it step by step.

**What you'll learn**
1. What ensemble learning is, and why it works
2. Bagging vs Boosting (the core difference)
3. How to apply both with scikit-learn, using real code
4. My actual results
5. Advantages and disadvantages
6. Mistakes in my own notebook you should avoid
7. When to use which

---

## 1. What is Ensemble Learning?

Ensemble learning combines **several models (weak learners)** to make one stronger prediction. Instead of trusting one Decision Tree, we train many and combine their answers (voting for classification, averaging for regression).

Why it works: individual models make *different* mistakes. When you combine them, many of those errors cancel out.

| Technique | Idea | How models are trained | Main goal |
|---|---|---|---|
| **Bagging** | Many models on random samples of the data | In **parallel**, independently | Reduce **variance** (overfitting) |
| **Boosting** | Each model fixes the previous one's mistakes | **Sequentially** | Reduce **bias** (underfitting) |
| Stacking / Voting | Combine different model types | Parallel + a meta-model | Mix strengths of different algorithms |

---

## 2. The Dataset

The Heart Failure Prediction dataset: **918 rows × 12 columns**.

| Feature | Meaning |
|---|---|
| Age, Sex | Patient basics |
| ChestPainType | ATA, NAP, ASY, TA |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| FastingBS | Fasting blood sugar > 120 (0/1) |
| RestingECG | Normal, ST, LVH |
| MaxHR | Max heart rate |
| ExerciseAngina | Y/N |
| Oldpeak | ST depression |
| ST_Slope | Up, Flat, Down |
| **HeartDisease** | **Target** (0 = no, 1 = yes) |

---

## 3. Preprocessing (do this before any ensemble)

### a) Fix impossible values
`Cholesterol = 0` and `RestingBP = 0` are not real measurements. They are missing data. In the notebook, 172 rows had Cholesterol = 0. I replaced zeros with the mean of the non-zero values:

```python
chol_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
df['Cholesterol'] = df['Cholesterol'].replace(0, chol_mean).round(2)

bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
df['RestingBP'] = df['RestingBP'].replace(0, bp_mean).round(2)
```

### b) Encode categorical columns
```python
df_encode = pd.get_dummies(df).astype(int)
```

### c) Scale numeric columns
```python
from sklearn.preprocessing import StandardScaler
cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR']
df_final = df_encode.copy()
df_final[cols] = StandardScaler().fit_transform(df_final[cols])
```

> Note: tree-based ensembles do **not** need scaling. Only Logistic Regression (our baseline) benefits from it.

### d) Train/test split
```python
from sklearn.model_selection import train_test_split

X = df_final.drop('HeartDisease', axis=1)
y = df_final['HeartDisease']          # a Series avoids the "column-vector y" warning

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 4. Baselines First

Always compare ensembles against simple models.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

lr = LogisticRegression().fit(X_train, y_train)
dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

print("LogReg  :", accuracy_score(y_test, lr.predict(X_test)))
print("DecTree :", accuracy_score(y_test, dt.predict(X_test)))
```

A single Decision Tree scored **~79%** on the test set and **100%** on the training set. That gap is classic **overfitting**: the tree memorized the training data. This is exactly the problem bagging solves.

---

## 5. Bagging (Bootstrap Aggregating)

### How it works
1. Draw many **bootstrap samples** (random rows, with replacement) from the training data.
2. Train one model on each sample, all independently.
3. Combine predictions by **majority vote**.

### Example 1: Random Forest (bagging + random feature selection)
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Train:", rf.score(X_train, y_train))
print("Test :", rf.score(X_test, y_test))
```
**Result: Train 1.00 | Test 0.880**

Random Forest is bagging of decision trees, with one extra trick: each split only considers a random subset of features, which makes the trees even more different from each other.

### Example 2: Generic BaggingClassifier
```python
from sklearn.ensemble import BaggingClassifier

bag = BaggingClassifier(
    estimator=RandomForestClassifier(),
    n_estimators=100,
    max_samples=0.8,      # each model sees 80% of the rows
    bootstrap=True,
    random_state=42
)
bag.fit(X_train, y_train)

print("Train:", bag.score(X_train, y_train))
print("Test :", bag.score(X_test, y_test))
```
**Result: Train 0.952 | Test 0.870**

`BaggingClassifier` works with **any** base model. The textbook version uses a plain Decision Tree:
```python
bag_tree = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100, max_samples=0.8, random_state=42
)
```

> 🧠 **Teaching note:** In my notebook I bagged a Random Forest, which is "bagging of bagging". It works, but it is heavy and rarely adds much over a single Random Forest. For learning, bag a plain Decision Tree so the effect is easy to see.

### Key bagging parameters
| Parameter | What it does |
|---|---|
| `n_estimators` | Number of models (more = more stable, slower) |
| `max_samples` | Fraction of rows per model |
| `max_features` | Fraction of columns per model |
| `bootstrap` | Sample with replacement or not |
| `oob_score=True` | Free validation using rows left out of each bootstrap |

---

## 6. Boosting

### How it works
1. Train a small, simple model (a shallow tree).
2. See where it made mistakes.
3. Train the next model to focus on those mistakes.
4. Repeat, adding each model's contribution with a small step (`learning_rate`).

### Example: Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_train, y_train)

print("Train:", gb.score(X_train, y_train))
print("Test :", gb.score(X_test, y_test))
```
**Result: Train 0.939 | Test 0.870**

### Other boosting options to try
```python
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier

ada  = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
hgb  = HistGradientBoostingClassifier(random_state=42)   # fast, handles big data
# Also popular: XGBoost, LightGBM, CatBoost
```

### Key boosting parameters
| Parameter | What it does |
|---|---|
| `n_estimators` | Number of boosting rounds |
| `learning_rate` | Step size. Lower = safer but needs more trees |
| `max_depth` | Depth of each tree (keep it small, 1 to 5) |
| `subsample` | Fraction of rows per round (< 1.0 adds regularization) |

`learning_rate` and `n_estimators` trade off: halve the learning rate, roughly double the trees.

---

## 7. Results Summary

| Model | Train acc. | Test acc. |
|---|---|---|
| Logistic Regression (baseline) | n/a | 0.870 |
| Decision Tree | 1.000 | 0.793 to 0.804 |
| Random Forest (bagging) | 1.000 | **0.880** |
| BaggingClassifier(RandomForest) | 0.952 | 0.870 |
| Gradient Boosting | 0.939 | 0.870 |

**What these numbers tell us**
- The ensembles lifted accuracy by roughly **7 to 8 points** over a single Decision Tree. That is the real win.
- Versus Logistic Regression, the gain is small. On a dataset this size, a simple linear model is already strong. Ensembles are not magic.
- The test set has only **184 rows**, so a 1% difference is about **2 patients**. Don't crown a winner from this alone. Use **cross-validation** (next section).

### Compare fairly with cross-validation
```python
from sklearn.model_selection import cross_val_score

models = {
    "LogReg": lr, "DecisionTree": dt, "RandomForest": rf,
    "Bagging": bag, "GradBoost": gb
}
for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=5)
    print(f"{name:13s} {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 8. Advantages vs Disadvantages

### Bagging
| ✅ Advantages | ❌ Disadvantages |
|---|---|
| Strongly reduces overfitting (variance) | Does little for **high-bias** (underfit) models |
| Training can run in parallel (`n_jobs=-1`) | Many models = more memory and slower prediction |
| Robust to noise and outliers | Less interpretable than one tree |
| Few parameters to tune, works well by default | Gains are small if the base model is already stable |
| OOB score gives a free validation estimate | |

### Boosting
| ✅ Advantages | ❌ Disadvantages |
|---|---|
| Often the highest accuracy on tabular data | **Sensitive to noisy labels and outliers** (keeps chasing them) |
| Reduces bias, can learn complex patterns | Sequential, so harder to parallelize and slower to train |
| Works with small, shallow trees | More hyperparameters to tune |
| Built-in feature importance | Can overfit if too many rounds or too high a learning rate |

### Ensembles in general
**Pros:** better accuracy, more stable predictions, less dependence on one lucky model.
**Cons:** more compute, harder to explain, and sometimes not worth it when a simple model is already good enough. In medical settings, explainability matters, so check feature importance:

```python
import pandas as pd
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(10))
```

---

## 9. Mistakes in My Notebook (so you can avoid them)

Teaching honestly means showing the flaws too:

1. **Data leakage from scaling.** I fitted `StandardScaler` on the *whole* dataset before splitting. Test-set statistics leaked into training. Fix: split first, then `fit` the scaler on `X_train` only, or use a `Pipeline`.
2. **`astype(int)` truncated real values.** Converting everything to int turned `Oldpeak` values like 1.5 into 1, and decimals in cholesterol/BP were cut. Fix: only convert the one-hot (boolean) columns, not numeric ones.
3. **Scaling a binary column.** `FastingBS` is 0/1 and shouldn't be standardized.
4. **Mean-imputing 172 cholesterol values** (19% of the data) can distort the distribution. Median imputation, or adding a "was missing" flag, is often better.
5. **Single train/test split.** With 184 test rows, results are noisy. Use cross-validation.
6. **Passing `y` as a DataFrame** caused the "column-vector y" warnings. Use a Series (or `.ravel()`).

The safest structure is a Pipeline:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

pre = ColumnTransformer([("scale", StandardScaler(), ['Age','RestingBP','Cholesterol','MaxHR'])],
                        remainder="passthrough")
pipe = Pipeline([("pre", pre), ("model", GradientBoostingClassifier(random_state=42))])
print(cross_val_score(pipe, X_raw, y, cv=5).mean())
```

---

## 10. Which One Should You Use?

| Situation | Try first |
|---|---|
| Model overfits (train ≫ test) | **Bagging / Random Forest** |
| Model underfits, want max accuracy | **Boosting** (XGBoost / LightGBM) |
| Noisy data with outliers | **Bagging** |
| Clean data, need best score | **Boosting** |
| Need a fast, strong default | **Random Forest** |
| Need explainability | Single tree or logistic regression, plus feature importance |

**Rule of thumb:** start with Random Forest as a baseline, then try Gradient Boosting and tune it. Always compare against a simple model before celebrating.

---

## 11. Try It Yourself (exercises)

1. Bag a plain `DecisionTreeClassifier` and compare it with the single tree.
2. Tune `n_estimators` and `learning_rate` of Gradient Boosting with `GridSearchCV`.
3. Try `AdaBoostClassifier` and `XGBClassifier`.
4. Build a `VotingClassifier` combining LogReg + RF + GB.
5. Fix the leakage issues from section 9 and see how the scores change.

---

## Quick Cheat Sheet

```text
Bagging  = parallel models + random samples  → less overfitting
Boosting = sequential models fixing errors   → less underfitting
Random Forest = bagging + random features
Always: split first → cross-validate → compare to a simple baseline
```

⭐ If this helped, star the repo and share it with someone learning ML.

*Tools: Python, pandas, scikit-learn, seaborn. Dataset: Heart Failure Prediction.*

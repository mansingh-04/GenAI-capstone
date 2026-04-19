import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score

df = pd.read_csv("../data/preprocessed/cleaned_dataset.csv")

df["content"] = df["content"].fillna("")
df = df[df["content"].str.strip() != ""]

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

results = {}


for name, model in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15000,
            min_df=5,
            max_df=0.9
        )),
        ("clf", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    results[name] = f1

    print(f"{name} F1 Score: {f1:.4f}")


best_model_name = max(results, key=results.get)
print(f"Best Base Model: {best_model_name}")

if best_model_name == "Logistic Regression":
    param_grid = {"clf__C": [0.1, 1, 10]}
    base_model = LogisticRegression(max_iter=1000, solver="liblinear")

elif best_model_name == "Linear SVM":
    param_grid = {"clf__C": [0.1, 1, 10]}
    base_model = LinearSVC()

else:
    param_grid = {"clf__alpha": [0.1, 1.0, 5.0]}
    base_model = MultinomialNB()

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        min_df=5,
        max_df=0.9
    )),
    ("clf", base_model)
])

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=1,
    verbose=1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation F1 Score:", grid.best_score_)

best_model = grid.best_estimator_


y_pred = best_model.predict(X_test)

print("Final Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


joblib.dump(best_model, "../news_credibility_model.pkl")
print("Final Model Saved Successfully.")

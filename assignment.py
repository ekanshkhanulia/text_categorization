# assignment.py
# Tasks 1–3: Load data, run 3 classifiers × 3 features, print macro precision, recall, and F1


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer # features

#three classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

#evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


RANDOM_STATE = 42

# Load all the 20 categories of  newsgroups dataset 
def load_data():
    train = fetch_20newsgroups(subset="train", shuffle=True, random_state=RANDOM_STATE,remove=('headers','footers','quotes')) #shuffle for shuffling the dataset
    test = fetch_20newsgroups(subset="test", shuffle=True, random_state=RANDOM_STATE,remove=('headers','footers','quotes'))
    return train.data, train.target, test.data, test.target, train.target_names

# Feature extractors 
def vectorizers():
    return {
        "Count": CountVectorizer(),
        "TF": Pipeline([
            ("count", CountVectorizer()),
            ("tf", TfidfTransformer(use_idf=False))
        ]),
        "TF-IDF": TfidfVectorizer()
    }

# Classifiers 
def classifiers():
    return {
        "NaiveBayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=2000),
        "LinearSVC": LinearSVC()
    }

#  Evaluate each pipeline: return macro-averaged Precision/Recall/F1  score
def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    return round(macro_precision, 3), round(macro_recall, 3), round(macro_f1, 3)

# Run 3x3 experiment (3 classifiers × 3 feature types)
def run_experiments(X_train, y_train, X_test, y_test):
    vecs = vectorizers()
    clfs = classifiers()
    results = []

    print("\n Running 3x3 Classifier × Feature Combinations ")
    for clf_name, clf in clfs.items():
        for vec_name, vec in vecs.items():
            pipe = Pipeline([
                ("vec", vec),
                ("clf", clf)
            ])
            precision, recall, f1 = evaluate_model(pipe, X_train, y_train, X_test, y_test)
            results.append({
                "Classifier": clf_name,
                "Feature": vec_name,
                "Macro Precision": precision,
                "Macro Recall": recall,
                "Macro F1": f1
            })
            print(f"{clf_name} + {vec_name} → Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    df = pd.DataFrame(results)

    print("\n=== Final Summary Table (Macro-F1 Scores) ===")
    summary_table = df.pivot(index="Classifier", columns="Feature", values="Macro F1")
    print(summary_table.round(3))

    # Save results for report
    df.to_csv("task3_full_metrics.csv", index=False)
    summary_table.to_csv("task3_macro_f1_summary.csv")

    return df, summary_table

# Task 4
def task4():
    {}

# Main 
def main():
    print("Loading 20 Newsgroups data...")
    X_train, y_train, X_test, y_test, target_names = load_data()

    print("Starting Task 2 + Task 3 Experiments...")
    run_experiments(X_train, y_train, X_test, y_test)

    
    task4()

if __name__ == "__main__":
    main()

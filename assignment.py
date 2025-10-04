
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# Three classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

RANDOM_STATE = 42

# Task 1 
# Load full 20-category dataset 
def load_data():
    return fetch_20newsgroups(subset="train", shuffle=True, random_state=RANDOM_STATE, remove=('headers','footers','quotes')), \
           fetch_20newsgroups(subset="test", shuffle=True, random_state=RANDOM_STATE, remove=('headers','footers','quotes'))

#  Task 2 
def run_task_2():
    print("\n Task 2: Comparing 3 Classifiers using TF-IDF  ")

    train, test = load_data()
    y_train = train.target
    y_test = test.target

    # TF-IDF vectorizer settings 
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_df=0.5,
        min_df=5,
        stop_words="english"
    )

    t0 = time()
    X_train = vectorizer.fit_transform(train.data)
    X_test = vectorizer.transform(test.data)
    print(f"TF-IDF vectorization  in {time() - t0:.2f}s")

    classifiers = {
        "NaiveBayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=2000),
        "LinearSVC": LinearSVC()
    }

    results = []
    print("\nClassifier Results (Macro Precision / Recall / F1):")
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"{name}  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        results.append({
            "Classifier": name,
            "Macro Precision": round(precision, 3),
            "Macro Recall": round(recall, 3),
            "Macro F1": round(f1, 3)
        })

    #  Save results
    df = pd.DataFrame(results)
    df.to_csv("task2_classifier_results.csv", index=False)

#  Task 3 
# Feature extractors 
def vectorizers():
    return {
        "Count": CountVectorizer(),
        "TF": Pipeline([
            ("count", CountVectorizer()),
            ("tf", TfidfTransformer(use_idf=False))
        ]),
        "TF-IDF": TfidfVectorizer(
            sublinear_tf=True,
            max_df=0.5,
            min_df=5,
            stop_words="english"
        )
        
    }

# Classifiers 
def classifiers():
    return {
        "NaiveBayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=2000),
        "LinearSVC": LinearSVC()
    }

# Evaluate each pipeline: return macro-averaged scores
def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    return round(macro_precision, 3), round(macro_recall, 3), round(macro_f1, 3)

# Task 3: Run 3x3 experiment (3 classifiers × 3 features)
def run_task_3():
    print("\n Task 3: 3 Classifiers × 3 Feature Types")
    train, test = load_data()
    X_train_raw, y_train = train.data, train.target
    X_test_raw, y_test = test.data, test.target

    vecs = vectorizers()
    clfs = classifiers()
    results = []

    for clf_name, clf in clfs.items():
        for vec_name, vec in vecs.items():
            pipe = Pipeline([
                ("vec", vec),
                ("clf", clf)
            ])
            precision, recall, f1 = evaluate_model(pipe, X_train_raw, y_train, X_test_raw, y_test)
            results.append({
                "Classifier": clf_name,
                "Feature": vec_name,
                "Macro Precision": precision,
                "Macro Recall": recall,
                "Macro F1": f1
            })
            print(f"{clf_name} + {vec_name} → Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    df = pd.DataFrame(results)
    print("\n Final Summary Table (Macro-F1 Scores) ")
    summary_table = df.pivot(index="Classifier", columns="Feature", values="Macro F1")
    print(summary_table.round(3))

    df.to_csv("task3_full_metrics.csv", index=False)
    summary_table.to_csv("task3_macro_f1_summary.csv")

#  Task 4
def run_task_4():
    print("\nTask 4")

#  Main Execution 
def main():
    run_task_2()
    run_task_3()
    run_task_4()

if __name__ == "__main__":
    main()
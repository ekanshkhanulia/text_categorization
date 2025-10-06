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
def run_task_1():
    data = fetch_20newsgroups(subset='all')
    categories = data.target_names

    print("\n Task 1: All 20 Newsgroup Categories\n")
    for i, cat in enumerate(categories):
        print(f"{i+1:2d}. {cat}")

    return categories

#  Data Loader 
def load_data(categories=None, verbose=False, remove=()):#'headers', 'footers', 'quotes'
    return fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=RANDOM_STATE, remove=remove), \
           fetch_20newsgroups(subset="test",  categories=categories, shuffle=True, random_state=RANDOM_STATE, remove=remove)

#  Task 2 
def run_task_2(categories):
    print("\n Task 2: Comparing 3 Classifiers using TF-IDF")

    train, test = load_data(categories)
    y_train = train.target
    y_test = test.target

    # TF-IDF vectorizer 
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

    df = pd.DataFrame(results)
    df.to_csv("task2_classifier_results.csv", index=False)

# Task 3 
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

def classifiers():
    return {
        "NaiveBayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=2000),
        "LinearSVC": LinearSVC()
    }

def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    return round(macro_precision, 3), round(macro_recall, 3), round(macro_f1, 3)

def run_task_3(categories):
    print("\n Task 3: 3 Classifiers × 3 Feature Types")
    train, test = load_data(categories)
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
            print(f"{clf_name} + {vec_name}  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    df = pd.DataFrame(results)
    print("\n Final Summary Table (Macro-F1 Scores) ")
    summary_table = df.pivot(index="Classifier", columns="Feature", values="Macro F1")
    print(summary_table.round(3))

    df.to_csv("task3_full_metrics.csv", index=False)
    summary_table.to_csv("task3_macro_f1_summary.csv")

#  Task 4 
def run_task_4(best_clf, X_train, y_train, X_test, y_test):
    """
    Run experiments with different TfidfVectorizer parameters
    using the best classifier found in Task 3.
    """
    print("\n Task 4: TfidfVectorizer Parameter Experiments\n")
    # Parameter grids for Task 4
    param_grid = [
        # Lowercasing
        {"lowercase": True},
        {"lowercase": False},

        # Stop words
        {"stop_words": None},
        {"stop_words": "english"},

        # N-gram and analyzer settings
        {"ngram_range": (1, 1)},        # unigrams
        {"ngram_range": (1, 2)},        # unigrams + bigrams
        {"ngram_range": (1, 3)},        # unigrams + bigrams + trigrams
        {"analyzer": "char", "ngram_range": (3, 5)},  # character 3–5-grams
        {"analyzer": "char", "ngram_range": (2, 4)},  # character 2–4-grams

        # Vocabulary size (feature limits)
        {"max_features": None},
        {"max_features": 3000},
        {"max_features": 5000},
    ]

    results = []

    print("\n=== Running TfidfVectorizer Parameter Experiments (Task 4) ===")
    for params in param_grid: # loop over each param setting
        # Build kwargs for TfidfVectorizer
        vec_kwargs = {"ngram_range": (1,1)}  # default
        vec_kwargs.update(params)            # override with test param(s)

        # Build pipeline
        pipe = Pipeline([
            ("vec", TfidfVectorizer(**vec_kwargs)),
            ("clf", best_clf)
        ])

        # Train & predict
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Metrics
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # Store results
        results.append({
            "Parameter": ",".join(params.keys()),
            "Value": ",".join([str(v) for v in params.values()]),
            "Macro Precision": round(precision, 3),
            "Macro Recall": round(recall, 3),
            "Macro F1": round(f1, 3)
        })


        print(f"Params {params} → F1={f1:.3f}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("task4_vectorizer_experiments.csv", index=False)

    print("\n=== Summary Table (All Results) ===")
    print(df.to_string(index=False))

    return df


def main():
    categories = run_task_1()
    #run_task_2(categories)
    #run_task_3(categories)
    best_clf = LinearSVC()  # From Task 3 results, LinearSVC was best
    train, test = load_data(categories)
    X_train, y_train = train.data, train.target
    X_test, y_test = test.data, test.target
    run_task_4(best_clf, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
   main()
   # test comment
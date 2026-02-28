from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from common import ensure_outputs_dir, load_dataset, write_json


def main() -> None:
    outputs_dir = ensure_outputs_dir()
    df = load_dataset()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(test_df["text"])
    similarity_matrix = cosine_similarity(x_test, x_train)
    max_similarity = similarity_matrix.max(axis=1)

    overlap_stats = {
        "mean": float(max_similarity.mean()),
        "std": float(max_similarity.std()),
        "min": float(max_similarity.min()),
        "max": float(max_similarity.max()),
        "q25": float(np.quantile(max_similarity, 0.25)),
        "q50": float(np.quantile(max_similarity, 0.50)),
        "q75": float(np.quantile(max_similarity, 0.75)),
    }
    write_json(overlap_stats, outputs_dir / "lexical_overlap_stats.json")

    baseline_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    x = df["text"].astype(str).to_numpy()
    y = df["label"].astype(str).to_numpy()
    scores = cross_val_score(baseline_pipeline, x, y, cv=cv, scoring="f1_macro")

    baseline_stats = {
        "f1_macro_mean": float(scores.mean()),
        "f1_macro_std": float(scores.std()),
        "fold_scores": [float(value) for value in scores.tolist()],
    }
    write_json(baseline_stats, outputs_dir / "baseline_cv_stats.json")


if __name__ == "__main__":
    main()

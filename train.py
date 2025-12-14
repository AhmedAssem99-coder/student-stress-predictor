# train.py (with GridSearchCV + metadata + CM image)
import json
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt

from prepare import (
    load_data,
    clean_data,
    split_features_target,
    build_preprocessor,
    split_data,
)

SEED = 42

def main():
    # 0) تجهيز مجلد الموديلات
    Path("models").mkdir(exist_ok=True)

    # 1) تحميل وتنظيف الداتا
    df = load_data("data/StressLevelDataset.csv")
    df = clean_data(df)
    X, y, num_cols, cat_cols = split_features_target(df)

    # 2) البري-بروسسور والموديل الأساسي
    preprocessor = build_preprocessor(num_cols, cat_cols)
    base_model = RandomForestClassifier(random_state=SEED, n_jobs=-1)

    pipe = Pipeline([("prep", preprocessor), ("model", base_model)])

    # 3) التقسيم
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4) GridSearchCV — Fine Tuning
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 6, 10, 14],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"]
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    print(">>> Running GridSearchCV...")
    grid.fit(X_train, y_train)
    print(f">>> Best params: {grid.best_params_}")
    print(f">>> Best CV accuracy: {grid.best_score_:.4f}")

    # 5) تقييم على Test
    best_pipe = grid.best_estimator_
    y_pred = best_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("\n=== Test Accuracy ===")
    print(f"{test_acc:.4f}")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 6) حفظ Confusion Matrix كصورة
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.close(fig)

    # 7) أهمية الميزات (Permutation Importance) على Train
    # نستخدم البري-بروسسور المثبت داخل best_pipe مباشرة
    prep_fitted = best_pipe.named_steps["prep"]
    Xt = prep_fitted.transform(X_train)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    imp = permutation_importance(
        best_pipe.named_steps["model"],
        Xt,
        y_train,
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1
    )

    feature_names = num_cols
    fi = pd.DataFrame({"feature": feature_names, "importance": imp.importances_mean})
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv("models/feature_importance.csv", index=False)

    # 8) حفظ الموديل
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_file = f"models/stress_model_{timestamp}.joblib"
    joblib.dump(best_pipe, model_file)
    print(f"Model saved to {model_file}")

    # 9) حفظ Metadata
    meta = {
        "model_file": Path(model_file).name,
        "best_params": grid.best_params_,
        "best_cv_accuracy": grid.best_score_,
        "test_accuracy": test_acc,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": num_cols,
        "timestamp_utc": timestamp
    }
    with open("models/metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Metadata saved to models/metadata.json")

if __name__ == "__main__":
    main()

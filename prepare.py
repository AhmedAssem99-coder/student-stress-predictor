# prepare.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

SEED = 42

SELECTED_FEATURES = [
    "anxiety_level",
    "depression",
    "sleep_quality",
    "academic_performance",
    "social_support",
]

TARGET_COL = "stress_level"

def load_data(path="data/StressLevelDataset.csv"):
    # نحمّل فقط الداتا الأساسية لأن فيها الهدف stress_level بشكل صريح
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # شيل الدوبلكيتس
    df = df.drop_duplicates()

    # تأكد إن الأعمدة المطلوبة موجودة
    missing = [c for c in SELECTED_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # معالجة القيم الناقصة للأعمدة الرقمية
    for c in SELECTED_FEATURES + [TARGET_COL]:
        if df[c].isna().any():
            # لو هدف فيه NaN، بنشيل الصفوف دي
            if c == TARGET_COL:
                df = df.dropna(subset=[TARGET_COL])
            else:
                df[c] = df[c].fillna(df[c].median())

    # تأكد من أن الهدف نوعه تصنيفي/أعداد صحيحة
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df

def split_features_target(df):
    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_COL].copy()

    # كل الأعمدة هنا رقمية، مفيش فئات نصية
    num_cols = SELECTED_FEATURES
    cat_cols = []

    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    # مفيش OneHotEncoder لأننا اخترنا أعمدة رقمية فقط
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols)
        ],
        remainder="drop"
    )
    return preprocessor

def split_data(X, y, test_size=0.2, seed=SEED):
    # لو التوزيع يَسمح، استخدم stratify
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:
        # لو مفيش كفاية فئات، نعمل split عادي
        return train_test_split(X, y, test_size=test_size, random_state=seed)

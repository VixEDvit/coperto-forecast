
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import FEATURE_COLUMNS

RANDOM_STATE = 42
MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("models/metrics.json")
FINAL_MODEL = "linear"  # выбрана по backtest, см. notebooks/02_modeling.ipynb

CATEGORICAL = ["dow", "month"]                  # one-hot: у дней недели нет «порядка»
FLAGS = ["is_weekend", "is_holiday", "is_jan1"]  # уже 0/1, передаём как есть
LEVELS = ["lag_7", "lag_14", "lag_21", "lag_28", "roll_mean_28"]  # в гостях -> log
SPREAD = ["roll_std_28"]


def build_linear_model() -> TransformedTargetRegressor:
    preprocess = ColumnTransformer([
        ("categories", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ("flags", "passthrough", FLAGS),
        ("levels", make_pipeline(SimpleImputer(strategy="median"),
                                 FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                 StandardScaler()), LEVELS),
        ("spread", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), SPREAD),
    ])
    pipeline = Pipeline([("preprocess", preprocess), ("ridge", Ridge(alpha=1.0))])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def build_boosting_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",      # оптимизируем то же, что меряем, — MAE
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        categorical_features=[FEATURE_COLUMNS.index(c) for c in CATEGORICAL],
        random_state=RANDOM_STATE,
    )


MODELS = {"linear": build_linear_model, "boosting": build_boosting_model}


def train(model_name: str, train_rows: pd.DataFrame, target: str = "guests"):
    """Обучает модель по имени на готовых строках с признаками."""
    if model_name not in MODELS:
        raise ValueError(f"Неизвестная модель {model_name!r}, есть: {sorted(MODELS)}")
    model = MODELS[model_name]()
    model.fit(train_rows[FEATURE_COLUMNS], train_rows[target])
    return model


def save_model(model, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Модель {path} не найдена. Обучите её: python -m src.model")
    return joblib.load(path)


if __name__ == "__main__":
    from src.data import prepare
    from src.features import make_features
    from src.validation import backtest, evaluable

    features = make_features(prepare("data/raw/guests.csv"))

    # Качество оцениваем до финального обучения: backtest на 4 периодах по 6 недель
    scores = backtest(features, [FINAL_MODEL])
    summary = scores.groupby("model")[["MAE", "MAPE, %"]].mean().round(2)
    metrics = {
        "model": FINAL_MODEL,
        "mae": float(summary.loc[FINAL_MODEL, "MAE"]),
        "mape": float(summary.loc[FINAL_MODEL, "MAPE, %"]),
        "baseline_mae": float(summary.loc["baseline", "MAE"]),
        "baseline_mape": float(summary.loc["baseline", "MAPE, %"]),
    }
    print(summary)

    # Финальная модель обучается на всей истории
    rows = evaluable(features)
    save_model(train(FINAL_MODEL, rows))
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Модель обучена на {len(rows)} днях и сохранена в {MODEL_PATH}")
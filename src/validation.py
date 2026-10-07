"""
Все признаки сдвинуты минимум на 7 дней, поэтому прогноз
на любой день отложенного периода использует только данные, известные за неделю до этого дня.
"""

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS


def time_split(df: pd.DataFrame, valid_weeks: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Последние valid_weeks недель валидация, всё что раньше обучение."""
    cutoff = df["date"].max() - pd.Timedelta(weeks=valid_weeks)
    train = df[df["date"] <= cutoff]
    valid = df[df["date"] > cutoff]
    if train.empty or valid.empty:
        raise ValueError("Недостаточно истории для разбиения по времени")
    return train, valid


def evaluable(df: pd.DataFrame, target: str = "guests") -> pd.DataFrame:
    mask = (df["is_closed"] == 0) & df[target].notna()
    return df[mask]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Средняя абсолютная ошибка в гостях."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Средняя абсолютная процентная ошибка по дням с y > 0"""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not mask.any():
        raise ValueError("Нет дней с ненулевым фактом: MAPE не определена")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


def baseline_predict(features: pd.DataFrame) -> pd.Series:
    return (features["lag_7"]
            .fillna(features["lag_14"])
            .fillna(features["roll_mean_28"]))


def compare(y_true: pd.Series, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Таблица MAE и MAPE для нескольких моделей"""
    rows = [{"model": name, "MAE": mae(y_true, pred), "MAPE, %": mape(y_true, pred)}
            for name, pred in predictions.items()]
    return pd.DataFrame(rows).set_index("model").round(2)


def backtest(features: pd.DataFrame, model_names: list[str],
             n_folds: int = 4, fold_weeks: int = 6) -> pd.DataFrame:
    from src.model import train  # импорт тут чтобы модули не зависели друг от друга по кругу

    end = features["date"].max()
    rows = []
    for k in range(n_folds):
        cutoff = end - pd.Timedelta(weeks=fold_weeks * (k + 1))
        stop = cutoff + pd.Timedelta(weeks=fold_weeks)
        train_rows = evaluable(features[features["date"] <= cutoff])
        valid_rows = evaluable(features[(features["date"] > cutoff) & (features["date"] <= stop)])

        predictions = {"baseline": baseline_predict(valid_rows)}
        for name in model_names:
            model = train(name, train_rows)
            predictions[name] = model.predict(valid_rows[FEATURE_COLUMNS])

        fold = compare(valid_rows["guests"], predictions).reset_index()
        fold.insert(0, "valid_from", (cutoff + pd.Timedelta(days=1)).date())
        rows.append(fold)
    return pd.concat(rows, ignore_index=True)
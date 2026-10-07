"""Генерация признаков. Одна и та же функция используется при обучении и при прогнозе.

Прогноз строится на HORIZON = 7 дней вперёд одной моделью, поэтому каждый
признак строки за день t вычисляется только по данным, известным в момент
t - HORIZON. Тогда все 7 дней прогноза можно посчитать без рекурсии.

"""

import pandas as pd

from src.holidays import is_holiday

HORIZON = 7
LAGS = (7, 14, 21, 28)
ROLLING_WINDOW = 28

FEATURE_COLUMNS = [
    "dow", "month", "is_weekend", "is_holiday", "is_jan1",
    *[f"lag_{lag}" for lag in LAGS],
    "roll_mean_28", "roll_std_28",
]


def check_calendar(df: pd.DataFrame) -> None:
    gaps = df.groupby("restaurant_id")["date"].diff().dropna() != pd.Timedelta(days=1)
    if gaps.any():
        raise ValueError("В календаре есть пропуски или дубли дат: "
                         "сначала вызовите restore_calendar из src.data")


def make_features(df: pd.DataFrame, target: str = "guests") -> pd.DataFrame:
    out = df.sort_values(["restaurant_id", "date"]).reset_index(drop=True)
    check_calendar(out)

    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["is_weekend"] = out["dow"].isin((4, 5, 6)).astype(int)  
    out["is_holiday"] = is_holiday(out["date"])
    out["is_jan1"] = ((out["date"].dt.month == 1) & (out["date"].dt.day == 1)).astype(int)


    observed = out[target].where(out["is_closed"] == 0)
    by_restaurant = observed.groupby(out["restaurant_id"])

    for lag in LAGS:
        out[f"lag_{lag}"] = by_restaurant.shift(lag)

    shifted = by_restaurant.shift(HORIZON)
    by_restaurant_shifted = shifted.groupby(out["restaurant_id"])
    out["roll_mean_28"] = by_restaurant_shifted.transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=14).mean())
    out["roll_std_28"] = by_restaurant_shifted.transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=14).std())

    return out
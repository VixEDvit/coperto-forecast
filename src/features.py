"""Генерация признаков. Одна и та же функция используется при обучении и при прогнозе.

Прогноз строится на HORIZON = 7 дней вперёд одной моделью, поэтому каждый
признак строки за день t вычисляется только по данным, известным в момент
t - HORIZON. Тогда все 7 дней прогноза можно посчитать сразу, без рекурсии.

Признаки:
    календарь   — dow, month, is_weekend, is_holiday, is_jan1 (известен заранее)
    лаги        — lag_7, lag_14, lag_21, lag_28 (пики автокорреляции в EDA)
    уровень     — roll_mean_28, roll_std_28: среднее и разброс за 4 недели,
                  окно заканчивается за HORIZON дней до t
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
    """Проверяет, что у каждого ресторана нет пропущенных дат.

    Иначе shift(7) сдвинет на 7 строк, а не на 7 дней, и лаги молча станут неверными.
    """
    gaps = df.groupby("restaurant_id")["date"].diff().dropna() != pd.Timedelta(days=1)
    if gaps.any():
        raise ValueError("В календаре есть пропуски или дубли дат: "
                         "сначала вызовите restore_calendar из src.data")


def make_features(df: pd.DataFrame, target: str = "guests") -> pd.DataFrame:
    """Добавляет признаки к таблице после src.data.prepare (или с будущими датами).

    Закрытые дни, сбои и выбросы в лагах и скользящих — NaN, а не 0:
    ноль в санитарный день ничего не говорит о спросе.
    """
    out = df.sort_values(["restaurant_id", "date"]).reset_index(drop=True)
    check_calendar(out)

    # --- календарь: известен заранее для любой даты ---
    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["is_weekend"] = out["dow"].isin((4, 5, 6)).astype(int)  # пт по загрузке как выходной
    out["is_holiday"] = is_holiday(out["date"])
    out["is_jan1"] = ((out["date"].dt.month == 1) & (out["date"].dt.day == 1)).astype(int)

    # --- наблюдаемый спрос: закрытые дни не несут информации о спросе ---
    observed = out[target].where(out["is_closed"] == 0)
    by_restaurant = observed.groupby(out["restaurant_id"])

    # --- лаги: значение k дней назад, k >= HORIZON ---
    for lag in LAGS:
        out[f"lag_{lag}"] = by_restaurant.shift(lag)

    # --- уровень: окно из 28 дней, заканчивается за HORIZON дней до t ---
    shifted = by_restaurant.shift(HORIZON)
    by_restaurant_shifted = shifted.groupby(out["restaurant_id"])
    out["roll_mean_28"] = by_restaurant_shifted.transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=14).mean())
    out["roll_std_28"] = by_restaurant_shifted.transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=14).std())

    return out
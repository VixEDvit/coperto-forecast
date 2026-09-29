"""Прогноз числа гостей на 7 дней вперёд.

Пример:
    python predict.py --date 2026-10-01 --restaurant 1
    python predict.py --date 2026-10-01 --restaurant 2 --output forecast.csv

--date — первый день прогноза. Используется только история ДО этой даты,
признаки считаются той же функцией, что и при обучении (src.features.make_features),
модель загружается из файла, а не обучается заново.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.data import prepare
from src.features import FEATURE_COLUMNS, HORIZON, ROLLING_WINDOW, make_features
from src.model import METRICS_PATH, load_model

DATA_PATH = "data/raw/guests.csv"
DAY_NAMES = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
MIN_HISTORY_DAYS = ROLLING_WINDOW + HORIZON  # чтобы посчитать уровень за 4 недели


def scheduled_closed(dates: pd.Series) -> pd.Series:
    """График работы: санитарный день — первый понедельник месяца.

    График известен заранее (как и календарь праздников), поэтому его можно
    использовать для будущих дат. В реальной системе он приходил бы от ресторана.
    """
    return ((dates.dt.dayofweek == 0) & (dates.dt.day <= 7)).astype(int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Прогноз числа гостей на 7 дней вперёд")
    parser.add_argument("--date", required=True, help="первый день прогноза, ГГГГ-ММ-ДД")
    parser.add_argument("--restaurant", required=True, type=int, help="идентификатор ресторана")
    parser.add_argument("--data", default=DATA_PATH, help="путь к сырым данным (CSV)")
    parser.add_argument("--output", help="сохранить прогноз в CSV по этому пути")
    return parser.parse_args()


def forecast(history: pd.DataFrame, start: pd.Timestamp, restaurant_id: int) -> pd.DataFrame:
    """Прогноз на HORIZON дней, начиная со start, по истории до start."""
    known = history[(history["restaurant_id"] == restaurant_id) & (history["date"] < start)]

    if known.empty:
        available = sorted(int(r) for r in history["restaurant_id"].unique())
        raise ValueError(f"Нет данных по ресторану {restaurant_id} до {start.date()}. "
                         f"Доступные рестораны: {available}")
    if len(known) < MIN_HISTORY_DAYS:
        raise ValueError(f"Слишком короткая история: {len(known)} дней до {start.date()}, "
                         f"нужно хотя бы {MIN_HISTORY_DAYS}")
    last_known = known["date"].max()
    if last_known < start - pd.Timedelta(days=1):
        raise ValueError(f"Данные заканчиваются {last_known.date()}: прогноз можно начать "
                         f"не позже {(last_known + pd.Timedelta(days=1)).date()}")

    future = pd.DataFrame({
        "date": pd.date_range(start, periods=HORIZON, freq="D"),
        "restaurant_id": restaurant_id,
        "guests": np.nan,  # будущее неизвестно
    })
    future["is_closed"] = scheduled_closed(future["date"])

    features = make_features(pd.concat([known, future], ignore_index=True))
    rows = features[features["date"] >= start]

    model = load_model()
    prediction = np.clip(model.predict(rows[FEATURE_COLUMNS]), 0, None).round()
    prediction[rows["is_closed"].to_numpy() == 1] = 0  # закрыто по графику

    return pd.DataFrame({
        "date": rows["date"].dt.date,
        "day": rows["date"].dt.dayofweek.map(dict(enumerate(DAY_NAMES))),
        "restaurant_id": restaurant_id,
        "guests_forecast": prediction.astype(int),
        "note": np.where(rows["is_closed"] == 1, "закрыто (санитарный день)", ""),
    })


def main() -> None:
    args = parse_args()
    try:
        start = pd.Timestamp(args.date)
    except ValueError:
        sys.exit(f"Ошибка: некорректная дата {args.date!r}, ожидается формат ГГГГ-ММ-ДД")

    try:
        result = forecast(prepare(args.data), start, args.restaurant)
    except (ValueError, FileNotFoundError) as error:
        sys.exit(f"Ошибка: {error}")

    print(f"Прогноз гостей, ресторан {args.restaurant}, "
          f"{result['date'].iloc[0]} — {result['date'].iloc[-1]}:\n")
    print(result.drop(columns="restaurant_id").to_string(index=False))

    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        print(f"\nКачество модели (backtest, 4 периода по 6 недель): "
              f"MAE {m['mae']} гостей, MAPE {m['mape']}% "
              f"(бейзлайн «неделю назад»: MAE {m['baseline_mae']}, MAPE {m['baseline_mape']}%)")

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nПрогноз сохранён в {args.output}")


if __name__ == "__main__":
    main()
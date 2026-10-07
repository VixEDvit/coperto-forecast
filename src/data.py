"""Загрузка и очистка данных.

Пустые дни бывают трёх видов, и они обрабатываются по-разному
закрыто (0 гостей, is_closed), сбой выгрузки (NaN, is_missing),
ресторан ещё не открылся (строк нет).
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_SCHEMA = {
    "date": "datetime64[ns]",  # день наблюдения, без времени
    "restaurant_id": "int64",  # идентификатор точки
    "guests": "int64",         # целевая переменная: число гостей за день
    "revenue": "float64",      # выручка за день, ₽
}

OUTLIER_RATIO = 2.5
SAME_DOW_WINDOW = 5  # сколько соседних одноимённых дней недели берём


def load_raw(path: str | Path) -> pd.DataFrame:
    """Читает сырые данные, проверяет колонки и приводит типы к RAW_SCHEMA."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Файл {path} не найден. Сгенерируйте его: python -m src.generate_data")

    df = pd.read_csv(path, parse_dates=["date"])

    missing = set(RAW_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"В данных нет колонок: {sorted(missing)}")

    df = df[list(RAW_SCHEMA)].astype(RAW_SCHEMA)

    # Одна и та же строка, выгруженная дважды дубликат, оставляем одну.
    df = df.drop_duplicates()

    # Два РАЗНЫХ значения за один день и ресторан это противоречие в данных.
    conflicts = df.duplicated(subset=["restaurant_id", "date"], keep=False)
    if conflicts.any():
        raise ValueError(
            f"Разные значения за одну дату: {df[conflicts].head().to_dict('records')}")

    return df.sort_values(["restaurant_id", "date"]).reset_index(drop=True)


def restore_calendar(df: pd.DataFrame) -> pd.DataFrame:
    bounds = df.groupby("restaurant_id")["date"].agg(["min", "max"])

    calendar = pd.concat([
        pd.DataFrame({
            "date": pd.date_range(row["min"], row["max"], freq="D"),
            "restaurant_id": restaurant_id,
        })
        for restaurant_id, row in bounds.iterrows()
    ])

    out = calendar.merge(df, on=["restaurant_id", "date"], how="left")
    # guests стал float64: в целых числах нет значения NaN («неизвестно»)
    out["is_missing"] = out["guests"].isna().astype(int)
    return out.reset_index(drop=True)


def mark_closed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_closed"] = (out["guests"] == 0).astype(int)
    return out


def mark_outliers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["guests_raw"] = out["guests"]

    # Закрытые дни не должны тянуть «норму» вниз
    open_guests = out["guests"].where(out["is_closed"] == 0)
    dow = out["date"].dt.dayofweek

    expected = open_guests.groupby([out["restaurant_id"], dow]).transform(
        lambda s: s.rolling(SAME_DOW_WINDOW, center=True, min_periods=3).median())

    ratio = open_guests / expected
    is_outlier = (ratio > OUTLIER_RATIO) | (ratio < 1 / OUTLIER_RATIO)
    out["is_outlier"] = is_outlier.astype(int)

    out.loc[is_outlier, ["guests", "revenue"]] = np.nan
    return out


def prepare(path: str | Path) -> pd.DataFrame:
    df = load_raw(path)
    df = restore_calendar(df)
    df = mark_closed(df)
    df = mark_outliers(df)
    return df


if __name__ == "__main__":
    clean = prepare("data/raw/guests.csv")
    summary = clean.groupby("restaurant_id")[
        ["is_missing", "is_closed", "is_outlier"]].sum()
    print(f"Строк после очистки: {len(clean)}")
    print(summary)
    print("\nНайденные выбросы:")
    print(clean.loc[clean["is_outlier"] == 1,
                    ["date", "restaurant_id", "guests_raw"]])
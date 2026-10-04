""" Генерация данных
Модель данных (для каждого ресторана):

    guests_t ~ Poisson(lambda_t)
    lambda_t = L(t) * W(dow_t) * S(t) * H(t)

    L(t)  — тренд (уровень ресторана, растёт со временем)
    W(d)  — недельная сезонность, среднее по неделе = 1
    S(t)  — годовая сезонность (летом веранда)
    H(t)  — праздники и сезон корпоративов
Запуск из корня проекта:
    python -m src.generate_data
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42  # фиксируем случайность: повторный запуск даёт тот же файл
END_DATE = "2026-09-30"
OUTPUT_PATH = Path("data/raw/guests.csv")

# Недельный профиль W(d): пн, вт, ср, чт, пт, сб, вс. Среднее ровно 1.
WEEKLY_PROFILE = np.array([0.80, 0.85, 0.90, 0.95, 1.25, 1.35, 0.90])

# Годовая сезонность S(t) = exp(A * cos(2*pi*(doy - peak) / 365.25))
YEARLY_AMPLITUDE = 0.12
YEARLY_PEAK_DAY = 196  # 15 июля — пик сезона веранд

# Праздники H(t): "месяц-день" -> множитель к ожидаемому числу гостей
HOLIDAYS = {
    "12-31": 1.8,  # Новый год: событие, а не выброс
    "01-01": 0.6,  # 1 января люди отсыпаются
    "01-02": 1.2, "01-03": 1.2, "01-04": 1.2, "01-05": 1.2,
    "01-06": 1.2, "01-07": 1.2, "01-08": 1.2,  # новогодние каникулы
    "02-14": 1.6,
    "02-23": 1.3,
    "03-08": 1.7,
    "05-01": 1.3,
    "05-09": 1.3,
    "06-12": 1.2,
    "11-04": 1.2,
}
CORPORATE_SEASON_MULTIPLIER = 1.25  # 15–30 декабря, корпоративы

# Параметры ресторанов. Ресторан 2 открылся позже то есть до открытия данных нет.
RESTAURANTS = [
    {"restaurant_id": 1, "open_date": "2024-10-01", "base_guests": 150,
     "yearly_growth": 0.12, "avg_check": 1500},
    {"restaurant_id": 2, "open_date": "2025-03-01", "base_guests": 90,
     "yearly_growth": 0.05, "avg_check": 1900},
]


def holiday_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """H(t): множитель праздников и сезона корпоративов для каждой даты."""
    month_day = pd.Series(dates.strftime("%m-%d"))
    h = month_day.map(HOLIDAYS).fillna(1.0).to_numpy()

    corporate = (dates.month == 12) & (dates.day >= 15) & (dates.day <= 30)
    return np.where(corporate, h * CORPORATE_SEASON_MULTIPLIER, h)


def expected_guests(dates: pd.DatetimeIndex, base_guests: float,
                    yearly_growth: float, is_new: bool) -> np.ndarray:
    """lambda_t = L(t) * W(dow_t) * S(t) * H(t) значит ожидаемое число гостей."""
    t_years = np.arange(len(dates)) / 365.25  # время с открытия, в годах

    level = base_guests * (1 + yearly_growth * t_years)  # L(t), линейный тренд
    if is_new:
        # новая точка «раскручивается»: первые месяцы гостей меньше
        days = np.arange(len(dates))
        level = level * (1 - 0.4 * np.exp(-days / 30))

    weekly = WEEKLY_PROFILE[dates.dayofweek]  # W(d)
    yearly = np.exp(YEARLY_AMPLITUDE * np.cos(
        2 * np.pi * (dates.dayofyear - YEARLY_PEAK_DAY) / 365.25))  # S(t)

    return level * weekly * yearly * holiday_multiplier(dates)


def simulate_restaurant(params: dict, is_new: bool,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Чистые данные одного ресторана."""
    dates = pd.date_range(params["open_date"], END_DATE, freq="D")
    lam = expected_guests(dates, params["base_guests"],
                          params["yearly_growth"], is_new)

    guests = rng.poisson(lam)  # шум: гости — счётная величина

    # Санитарный день: первый понедельник месяца, ресторан закрыт.
    # Это известно заранее (график работы), поэтому гостей честно 0.
    closed = (dates.dayofweek == 0) & (dates.day <= 7)
    guests[closed] = 0

    # Выручка = гости * средний чек; чек немного гуляет от дня ко дню
    check = params["avg_check"] * rng.lognormal(0, 0.05, size=len(dates))
    revenue = np.round(guests * check, 2)

    return pd.DataFrame({
        "date": dates,
        "restaurant_id": params["restaurant_id"],
        "guests": guests,
        "revenue": revenue,
    })


def corrupt(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Портим данные так, как их портит реальная выгрузка из кассы."""
    n = len(df)

    # 1. Сбой выгрузки: ~2% случайных дней и один блок из 5 дней подряд
    #    просто отсутствуют в файле (строки нет вообще).
    lost = rng.choice(n, size=int(0.02 * n), replace=False)
    block_start = rng.integers(60, n - 60)
    lost = np.union1d(lost, np.arange(block_start, block_start + 5))

    # 2. Выбросы: ошибка ввода, гости и выручка записаны в 3 раза больше.
    open_days = np.flatnonzero(df["guests"].to_numpy() > 0)
    spikes = rng.choice(np.setdiff1d(open_days, lost), size=3, replace=False)
    df = df.copy()
    df.loc[df.index[spikes], ["guests", "revenue"]] *= 3

    df = df.drop(index=df.index[lost])

    # 3. Дубликаты: одна и та же строка выгружена дважды.
    dup = df.sample(n=3, random_state=int(rng.integers(1_000_000)))
    return pd.concat([df, dup])


def main() -> None:
    rng = np.random.default_rng(SEED)

    frames = []
    for i, params in enumerate(RESTAURANTS):
        clean = simulate_restaurant(params, is_new=(i > 0), rng=rng)
        frames.append(corrupt(clean, rng))

    raw = pd.concat(frames).sort_values(["restaurant_id", "date"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUTPUT_PATH, index=False)
    print(f"Сохранено {len(raw)} строк в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
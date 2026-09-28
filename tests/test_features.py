"""Тесты признаков: главное — ни один признак не заглядывает в будущее.

Запуск из корня проекта:
    pytest
"""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, HORIZON, make_features


def make_history(n_days: int = 120) -> pd.DataFrame:
    """Маленький искусственный ряд одного ресторана для тестов."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n_days, freq="D"),
        "restaurant_id": 1,
        "guests": rng.poisson(150, size=n_days).astype(float),
        "is_closed": 0,
    })


def test_features_do_not_see_the_future():
    """Меняем всё, начиная с дня T. Признаки дней до T + HORIZON не должны измениться.

    Признак дня t может зависеть только от данных до t - HORIZON включительно.
    Значит, данные с дня T и позже влияют только на признаки дней >= T + HORIZON.
    """
    history = make_history()
    cutoff = history["date"].iloc[80]  # день T

    changed = history.copy()
    changed.loc[changed["date"] >= cutoff, "guests"] = 10_000  # «испорченное будущее»

    before = make_features(history)
    after = make_features(changed)

    unaffected = before["date"] < cutoff + pd.Timedelta(days=HORIZON)
    pd.testing.assert_frame_equal(
        before.loc[unaffected, FEATURE_COLUMNS],
        after.loc[unaffected, FEATURE_COLUMNS],
    )


def test_lag_7_is_value_one_week_ago():
    """lag_7 в день t равен числу гостей в день t - 7."""
    features = make_features(make_history())
    expected = features["guests"].shift(7)
    pd.testing.assert_series_equal(features["lag_7"], expected, check_names=False)


def test_target_is_not_a_feature():
    """Целевая переменная не должна попасть в список признаков."""
    assert "guests" not in FEATURE_COLUMNS


def test_closed_days_are_nan_in_lags():
    """Закрытый день даёт NaN в лаге, а не 0: ноль не говорит о спросе."""
    history = make_history()
    history.loc[10, ["guests", "is_closed"]] = [0, 1]
    features = make_features(history)
    assert np.isnan(features.loc[17, "lag_7"])


def test_missing_dates_raise_clear_error():
    """Пропущенная дата должна давать понятную ошибку, а не молча сдвигать лаги."""
    history = make_history().drop(index=50)
    with pytest.raises(ValueError, match="пропуски"):
        make_features(history)
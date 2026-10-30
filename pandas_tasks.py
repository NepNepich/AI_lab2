"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    info = pd.read_csv(data.csv_path)
    row_count = len(info)
    missing_by_column = info.isna().sum().to_dict()
    ages = info["Age"].dropna()
    adults_over_30_count = int((ages > 30).sum())
    mean_age_series = info.groupby("Pclass")["Age"].mean()
    mean_age_by_pclass = {int(k): v for k, v in mean_age_series.items()}
    survival_rate_series = info.groupby("Pclass")["Survived"].mean()
    survival_rate_by_pclass = {int(k): v for k, v in survival_rate_series.items()}
    highest_fares = info["Fare"].nlargest(5).tolist()
    
    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares
    )
    

def test_analyze_titanic():
    data = TitanicInput(csv_path="lab2/mini_titanic.csv")
    result = analyze_titanic(data)
    
    print("--- Результаты теста ---")
    
    # 1. Проверка количества строк
    assert result.row_count == 8, f"Ожидалось 8 строк, получено {result.row_count}"
    print("✅ row_count: 8")
    
    # 2. Проверка пропусков
    # В нашем файле пропуск только у Age у пассажира №6 (Moran)
    assert result.missing_by_column["Age"] == 1, f"Пропусков в Age должно быть 1, а не {result.missing_by_column["Age"]}"
    assert result.missing_by_column["Cabin"] == 5, f"Пропусков в Cabin должно быть 5"
    print("✅ missing_by_column корректно")
    
    # 3. Пассажиры старше 30 лет
    # 38 (Cumings), 35 (Futrelle), 35 (Allen), 54 (McCarthy). Итого 4.
    # Пассажир Moran (NaN) не учитывается.
    assert result.adults_over_30_count == 4, f"Ожидалось 4 человека > 30 лет, получено {result.adults_over_30_count}"
    print("✅ adults_over_30_count: 4")
    
    # 4. Средний возраст по классам
    # Pclass 1: Cumings(38), Futrelle(35), Palsson(2). Mean = (38+35+2)/3 = 25.0
    # Pclass 2: McCarthy(54). Mean = 54.0
    # Pclass 3: Braund(22), Heikkinen(26), Allen(35), Moran(NaN). Mean = (22+26+35)/3 = 27.666...
    
    expected_age_p1 = 25.0
    expected_age_p3 = (22 + 26 + 35) / 3
    
    assert abs(result.mean_age_by_pclass[1] - expected_age_p1) < 0.01, "Ошибка в среднем возрасте для 1 класса"
    assert abs(result.mean_age_by_pclass[3] - expected_age_p3) < 0.01, "Ошибка в среднем возрасте для 3 класса"
    print(f"✅ mean_age_by_pclass: {result.mean_age_by_pclass}")
    
    # 5. Доля выживших по классам
    # Pclass 1: Cumings(1), Futrelle(1), Palsson(0). Survived: [1, 1, 0]. Mean = 2/3 ≈ 0.666
    # Pclass 3: Braund(0), Heikkinen(1), Allen(0), Moran(0). Survived: [0, 1, 0, 0]. Mean = 1/4 = 0.25
    
    expected_survival_p1 = 2/3
    expected_survival_p3 = 0.25
    
    assert abs(result.survival_rate_by_pclass[1] - expected_survival_p1) < 0.01, "Ошибка в доле выживших для 1 класса"
    assert abs(result.survival_rate_by_pclass[3] - expected_survival_p3) < 0.01, "Ошибка в доле выживших для 3 класса"
    print(f"✅ survival_rate_by_pclass: {result.survival_rate_by_pclass}")
    
    # 6. Топ-5 тарифов
    # Fare: 7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075
    # Сортировка по убыванию: 71.2833, 53.1, 51.8625, 21.075, 8.4583
    expected_top_fares = [71.2833, 53.1, 51.8625, 21.075, 8.4583]
    
    assert result.highest_fares == expected_top_fares, f"Топ тарифов не совпадает. Ожидалось {expected_top_fares}, получено {result.highest_fares}"
    print(f"✅ highest_fares: {result.highest_fares}")

    print("\n🎉 Все тесты пройдены успешно!")

test_analyze_titanic()
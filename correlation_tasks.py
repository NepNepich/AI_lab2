"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd
from scipy import stats

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    info = pd.read_csv(data.csv_path, sep='\t', na_values=['NA'])
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    target = 'MRI_Count'
    males = info[info['Gender'] == 'Male'].dropna(subset=features + [target])
    females = info[info['Gender'] == 'Female'].dropna(subset=features + [target])
    men_count = len(males)
    women_count = len(females)
    men_mri_correlation = {}
    women_mri_correlation = {}
    max_abs_corr = -1.0
    strongest_feature = ""
    for feature in features:
        if men_count > 0:
            r_m, _ = stats.pearsonr(males[feature], males[target])
            men_mri_correlation[feature] = r_m
            if abs(r_m) > max_abs_corr:
                max_abs_corr = abs(r_m)
                strongest_feature = feature
                
        if women_count > 0:
            r_f, _ = stats.pearsonr(females[feature], females[target])
            women_mri_correlation[feature] = r_f
            if abs(r_f) > max_abs_corr:
                max_abs_corr = abs(r_f)
                strongest_feature = feature
                
    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_feature
    )

def test_analyze_brain_correlations():
    data = BrainDataInput(csv_path="brainsize.txt")
    result = analyze_brain_correlations(data)
    
    print("--- Результаты теста корреляций ---")
    
    # 1. Проверка количества наблюдений
    # В файле 40 строк данных. 
    # Женщин: 20, Мужчин: 20.
    # Но у мужчин есть 2 строки с NA (строки 2 и 21).
    # Если функция считает dropna, то валидных мужчин будет 18.
    # Обычно в таких задачах count считается по исходным группам до отсечения пар для корреляции.
    # Посмотрим, что вернет функция. В моем решении я считаю len(males) после dropna.
    # В мини-датасете: 20 женщин (все валидны по target+features), 18 мужчин (2 имеют NA).
    
    print(f"Men count: {result.men_count}")
    print(f"Women count: {result.women_count}")
    
    # 2. Проверка корреляций (примерные значения, рассчитанные через scipy)
    # Для женщин самая сильная корреляция обычно с Weight или Height.
    # Для мужчин тоже.
    
    print(f"Men correlations: {result.men_mri_correlation}")
    print(f"Women correlations: {result.women_mri_correlation}")
    
    # 3. Проверка strongest_mri_feature
    # Это имя признака с максимальным abs(r) среди всех признаков обеих групп.
    print(f"Strongest feature: {result.strongest_mri_feature}")
    
    # Простая проверка на тип данных
    assert isinstance(result.men_mri_correlation, dict), "Корреляции мужчин должны быть словарем"
    assert isinstance(result.strongest_mri_feature, str), "Имя признака должно быть строкой"
    
    # Проверка, что признак действительно из списка
    valid_features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    assert result.strongest_mri_feature in valid_features, f"Неверный признак: {result.strongest_mri_feature}"
    
    print("\n✅ Базовые проверки пройдены!")

test_analyze_brain_correlations()
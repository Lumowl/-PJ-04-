
import pandas as pd
import numpy as np
import re
import ast
import json
from collections import Counter

def _do_preprocessing(df):
    df = df.copy()

    # Сбрасываю индексы в начале
    df = df.reset_index(drop=True)

    # Удаляю ненужные модели параметры
    columns_to_drop = []
    for col in ['mls-id', 'MlsId']:
        if col in df.columns:
            columns_to_drop.append(col)
    if columns_to_drop:
        df.drop(columns_to_drop, axis=1, inplace=True)

    if 'PrivatePool' in df.columns and 'private pool' in df.columns:
        df['pool'] = df['PrivatePool'].combine_first(df['private pool'])
    elif 'PrivatePool' in df.columns:
        df['pool'] = df['PrivatePool']
    elif 'private pool' in df.columns:
        df['pool'] = df['private pool']
    else:
        df['pool'] = 'no'  # или pd.NA

    df['pool'] = df['pool'].fillna('no').str.lower().str.strip().eq('yes')

    # Удаляю старые признаки с бассейном - ТОЛЬКО ЕСЛИ ОНИ ЕСТЬ
    columns_to_drop = []
    for col in ['PrivatePool', 'private pool']:
        if col in df.columns:
            columns_to_drop.append(col)

    if columns_to_drop:
        df.drop(columns_to_drop, axis=1, inplace=True)

    # Разметка сокращенных статусов
    short_status_map = {
        "C": "Continue Show",
        "P": "Pending Sale",
        "U": "Under Contract"
    }

    def normalize_status(s):
        """Нормальзую признак статуса"""

        if pd.isna(s):
            return "missing"

        if s in short_status_map:
            return short_status_map[s]

        s = s.lower()

        if "missing" in s:
            return "missing"

        if "active" in s or "for sale" in s or "continue show" in s:
            return "active"

        if "pending" in s or "contract" in s or "option" in s:
            return "pending/under Contract"

        if "contingent" in s:
            return "contingent"

        if "auction" in s or "foreclos" in s or "pre-fore" in s:
            return "auction/foreclosure"

        if "new" in s or "coming" in s or "extended" in s or "price change" in s or "back on market" in s:
            return "new/coming Soon"

        if "sold" in s or "closed" in s:
            return "sold"

        if "rent" in s:
            return "rent"

        return "other"

    # Применяю
    df["status_cat"] = df["status"].apply(normalize_status)


    def normalize_property_type(s):
        """Нормализую признак типа постройки"""

        if pd.isna(s):
            return "Missing"
        s = s.lower().strip()

        if "missing" in s or s == "":
            return "Missing"

        # Single Family (самая широкая группа)
        if "single" in s or "detached" in s or "story" in s:
            return "single Family"
        if "traditional" in s or "colonial" in s or "craftsman" in s:
            return "single Family"
        if "ranch" in s or "bungalow" in s or "cape cod" in s:
            return "single Family"
        if "contemporary" in s or "modern" in s or "transitional" in s:
            return "single Family"

        # Condo
        if "condo" in s:
            return "condo"

        # Townhouse
        if "town" in s or "row home" in s:
            return "townhouse"

        # Multi-family
        if "multi" in s or "multiple occupancy" in s:
            return "multi-family"

        # Land
        if "land" in s or "lot" in s:
            return "land"

        # Apartment / Co-op
        if "apart" in s or "coop" in s or "cooperative" in s or "high rise" in s:
            return "apartment/co-Op"

        # Mobile
        if "mobile" in s or "manufact" in s or "mfd" in s:
            return "mobile/manufactured"

        # Farm
        if "farm" in s or "ranch" in s:
            return "farm/ranch"

        return "other"

    # Применяю
    df["propertyType_cat"] = df["propertyType"].apply(normalize_property_type)

    # Список значений, которые считаем как «неизвестный адрес»
    undisclosed = ["MISSING", "Address Not Disclosed", "Undisclosed Address",
                   "(undisclosed Address)", "Address Not Available", "Unknown Address"]

    def normalize_street(s):
        """Нормальзую признак улицы"""

        if pd.isna(s) or s in undisclosed:
            return "undisclosed"
        else:
            return "known"

    # Применяю
    df["street"] = df["street"].fillna("MISSING")

    df["street_cat"] = df["street"].apply(normalize_street)

    # Вычисляю моду среди нормальных значений (1–10)
    valid_baths = df["baths"].apply(lambda x: re.search(r"(\d+(\.\d+)?)", str(x).replace(",", "")))
    valid_baths = valid_baths[valid_baths.notnull()].apply(lambda m: float(m.group(1)))
    valid_baths = valid_baths[(valid_baths >= 1) & (valid_baths <= 10)]
    mode_baths = Counter(valid_baths).most_common(1)[0][0]

    def clean_baths(x, mode_val):
        """Очищаю признак кол-ва ван"""

        if pd.isna(x) or str(x).strip().lower() in ["missing", ""]:
            return 0
        x_str = str(x).lower().strip()
        match = re.search(r"(\d+(\.\d+)?)", x_str.replace(",", "."))
        if match:
            val = float(match.group(1))
            if val > 10:
                return mode_val
            return val
        return mode_val

    # Применяю
    df["baths_clean"] = df["baths"].apply(lambda x: clean_baths(x, mode_baths))

    def extract_home_facts(x):
        """Преобразую значение признака"""

        def clean_value(v):

            if v in [None, '', '—', 'No Data']:
                return 0

            # Если это строка: убираю пробелы и мусор
            if isinstance(v, str):
                v = v.strip()
                # Оставляю только цифры, если это похоже на число
                v_digits = re.sub(r'[^0-9]', '', v)
                if v_digits.isdigit():
                    return int(v_digits)
                return v  # Если нет цифр — оставить как есть (категориальное значение)

            return v  # Числа вернуть как есть

        result = {}
        try:
            data_list = ast.literal_eval(x)
            for fact in data_list.get('atAGlanceFacts', []):
                val = clean_value(fact.get('factValue'))
                label = fact.get('factLabel')
                result[label] = val

        except:
            result = {
                'Year built': 0,
                'Remodeled year': 0,
                'Heating': 0,
                'Cooling': 0,
                'Parking': 0,
                'lotsize': 0,
                'Price/Sqft': 0
            }

        return result

    # Применяю и разворачиваю в отдельные колонки
    home_facts_df = df['homeFacts'].apply(extract_home_facts).apply(pd.Series)

    home_facts_df.index = df.index
    # Объединяю с основным датафреймом
    df = pd.concat([df, home_facts_df], axis=1)

    df['Heating'] = df['Heating'].fillna('').astype(str)
    df['Heating'] = df['Heating'].str.replace(',', '', regex=False).str.lower()
    df['Heating'] = df['Heating'].str.strip()

    def categorize_heating(value):
        """Категоризирую признак отопления"""

        if pd.isna(value) or value == 'missing':
            return 'Missing'

        text = str(value).lower()

        # Основные 10 категорий
        if 'forced air' in text or 'forcedair' in text:
            return 'forced air'

        if 'heat pump' in text:
            return 'heatpump'

        if 'central' in text:
            return 'central'

        if 'electric' in text:
            return 'electric'

        if 'gas' in text or 'natural' in text:
            return 'gas'

        if 'baseboard' in text:
            return 'baseboard'

        if 'wall' in text and 'window' not in text:
            return 'wall heater'

        if 'radiant' in text or 'hot water' in text or 'steam' in text:
            return 'radiant/water'

        if text == 'other':
            return 'other'

        if 'none' in text or 'no cooling' in text:
            return 'none'

        return 'other'

    # Применяю
    df["heating_cat"] = df["Heating"].apply(categorize_heating)

    def categorize_cooling(value):
        """Категоризирую охлаждение"""

        if pd.isna(value) or value == 'missing' or str(value) == '0':
            return 'Missing'

        text = str(value).lower()

        # Основные категории охлаждения
        if 'central air' in text or 'central a/c' in text or 'air conditioning-central' in text:
            return 'central air'

        if 'central' in text and ('cooling' not in text and 'electric' not in text and 'gas' not in text):
            return 'central'

        if 'refrigeration' in text:
            return 'refrigeration'

        if 'evaporative' in text or 'swamp' in text:
            return 'evaporative'

        if 'heat pump' in text:
            return 'heat pump'

        if 'window' in text or 'wall/window' in text or 'wall unit' in text:
            return 'window/wall unit'

        if 'electric' in text:
            return 'electric'

        if 'gas' in text:
            return 'gas'

        if 'none' in text or 'no heating' in text:
            return 'none'

        if 'other' in text:
            return 'other'

        if 'has cooling' in text or 'cooling system' in text:
            return 'has cooling'

        # Цифры и мусор
        if text in ['1', '2', '3', '90', 'has heating']:
            return 'other'

        # Всё остальное
        return 'other'

    # Применяю
    df['cooling_cat'] = df['Cooling'].apply(categorize_cooling)

    # Применяю
    def categorize_parking(value):
        """Категоризируем парковку"""

        if pd.isna(value) or str(value) == '0':
            return 'Missing'

        text = str(value).lower()

        # Основные типы парковки
        if 'attached garage' in text or 'garage-attached' in text or 'garage attached' in text:
            return 'attached garage'

        if 'detached garage' in text or 'detached parking' in text:
            return 'detached garage'

        if 'carport' in text:
            return 'carport'

        if 'off street' in text or 'offstreet' in text:
            return 'off street'

        if 'on street' in text or 'onstreet' in text:
            return 'on street'

        if 'driveway' in text:
            return 'driveway'

        if 'none' in text:
            return 'none'

        if text.isdigit():
            num = int(text)
            if num <= 0:
                return 'missing'
            elif num == 1:
                return '1 Space'
            elif num == 2:
                return '2 Spaces'
            elif num == 3:
                return '3 Spaces'
            elif num <= 6:
                return '4-6 Spaces'
            else:
                return '7+ Spaces'

        if 'parking' in text and ('desc' in text or 'type' in text or 'yn' in text):
            return 'Other Parking'

        # Всё остальное
        return 'other'

    # Применяю
    df['parking_cat'] = df['Parking'].apply(categorize_parking)

    def clean_lotsize(x):
        """Очищаю признак размера участка"""

        if pd.isna(x) or str(x).lower() in ["missing", "no data", "(other)"]:
            return 0
        x_str = str(x).lower().replace(",", "").strip()
        if "acre" in x_str:
            match = re.search(r"([\d\.]+)", x_str)
            if match:
                return int(float(match.group(1)) * 43560)
        elif "sqft" in x_str or "sq. ft." in x_str:
            match = re.search(r"([\d\.]+)", x_str)
            if match:
                return float(match.group(1))
        else:
            match = re.search(r"([\d\.]+)", x_str)
            if match:
                return float(match.group(1))
        return 0

    # Применяю
    df["lotsize_clean"] = df["lotsize"].apply(clean_lotsize)

    def categorize_lotsize(x):
        """Категоризирую очищенный размер участка"""
        if pd.isna(x):
            return "missing"
        elif x < 1500:
            return "urban_condo"
        elif x < 3000:
            return "urban_rowhouse"
        elif x < 5000:
            return "urban_small_lot"
        elif x < 7500:
            return "urban_standard"
        elif x < 10000:
            return "suburban_small"
        elif x < 21780:
            return "suburban_quarter"
        elif x < 43560:
            return "suburban_half"
        elif x < 108900:
            return "suburban_full"
        elif x < 217800:
            return "rural_small"
        else:
            return "rural_large"

    # Применяю
    df["lotsize_cat"] = df["lotsize_clean"].apply(categorize_lotsize)

    def parse_schools(schools_str):
        """Меняю значение признака школы"""

        if pd.isna(schools_str) or schools_str == '':
            return []

        # Если уже список
        if isinstance(schools_str, list):
            return schools_str

        # Пробую разные методы парсинга
        try:
            # Метод 1: ast.literal_eval
            parsed = ast.literal_eval(str(schools_str))
            if isinstance(parsed, list):
                return parsed
        except:
            try:
                # Метод 2: json с заменой None
                json_str = str(schools_str).replace("'", '"').replace('None', 'null')
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass

        return []

    # Применяю
    df['schools_parsed'] = df['schools'].apply(parse_schools)

    def extract_numeric_ratings(schools_list):
        """Извлекаю рейтинг школ"""

        if not isinstance(schools_list, list):
            return []

        numeric_ratings = []

        for school in schools_list:
            if not isinstance(school, dict):
                continue

            ratings = school.get('rating', [])
            if not isinstance(ratings, list):
                continue

            for rating_str in ratings:
                if not isinstance(rating_str, str):
                    continue

                # NR → 0
                if rating_str.strip().upper() == "NR":
                    numeric_ratings.append(0)
                    continue

                # Берём первое число
                match = re.search(r'\d+', rating_str)
                if match:
                    try:
                        numeric_ratings.append(int(match.group()))
                    except:
                        pass

        return numeric_ratings

    # Применяю и создаю признаки
    df['school_ratings_list'] = df['schools_parsed'].apply(extract_numeric_ratings)

    # Средний рейтинг
    df['avg_school_rating'] = df['school_ratings_list'].apply(
        lambda x: round(np.mean(x), 2) if x else 0
    )

    # Максимальный рейтинг
    df['max_school_rating'] = df['school_ratings_list'].apply(
        lambda x: max(x) if x else 0
    )

    # Количество школ с рейтингом >= 7/10
    df['num_good_schools'] = df['school_ratings_list'].apply(
        lambda x: sum(1 for r in x if r >= 7) if x else 0
    )

    def extract_distances(schools_list):
        """Извлекаю расстояние до школ"""

        if not isinstance(schools_list, list):
            return []

        distances = []
        for school in schools_list:
            if isinstance(school, dict):
                data_dict = school.get('data', {})
                if isinstance(data_dict, dict):
                    dist_list = data_dict.get('Distance', [])
                    if isinstance(dist_list, list):
                        for dist_str in dist_list:
                            if dist_str and isinstance(dist_str, str):
                                # Ищем число перед 'mi'
                                match = re.search(r'(\d+\.?\d*)', dist_str)
                                if match:
                                    try:
                                        distances.append(float(match.group(1)))
                                    except:
                                        pass

        return distances

    # Применяю и создаю признаки
    df['school_distances'] = df['schools_parsed'].apply(extract_distances)

    # Минимальное расстояние
    df['min_school_distance_mi'] = df['school_distances'].apply(
        lambda x: min(x) if x else 0
    )

    # Среднее расстояние
    df['avg_school_distance_mi'] = df['school_distances'].apply(
        lambda x: round(np.mean(x), 2) if x else 0
    )

    # Количество школ в пределах 1 мили
    df['schools_within_1mi'] = df['school_distances'].apply(
        lambda x: sum(1 for d in x if d <= 1) if x else 0
    )

    def analyze_grades(schools_list):
        """Разбираю уровни школ"""

        if not isinstance(schools_list, list):
            return {
                'elementary': False, 'middle': False, 'high': False,
                'special': False, 'grades_list': []
            }

        grades_found = []
        for school in schools_list:
            if isinstance(school, dict):
                data_dict = school.get('data', {})
                if isinstance(data_dict, dict):
                    grades_list = data_dict.get('Grades', [])
                    if isinstance(grades_list, list):
                        grades_found.extend([str(g).upper() for g in grades_list])

        # Анализирую найденные grades
        grades_str = ' '.join(grades_found)

        # Более точное определение
        has_elementary = any(g in grades_str for g in ['PK-5', 'K-5', 'PK-6', 'K-6', 'PK-8', '1-5', '1-6'])
        has_middle = any(g in grades_str for g in ['6-8', '7-8', '6-9', '5-8'])
        has_high = any(g in grades_str for g in ['9-12', '10-12', '9-10'])
        has_special = any(g in grades_str for g in ['K-9', 'K-12', 'PK-12', '6-12'])

        return {
            'has_elementary': has_elementary,
            'has_middle': has_middle,
            'has_high': has_high,
            'has_special': has_special,
            'grades_list': list(set(grades_found))
        }

    grades_info = df['schools_parsed'].apply(analyze_grades)

    # Создаю бинарные признаки
    df['has_elementary_school'] = grades_info.apply(lambda x: x['has_elementary'])
    df['has_middle_school'] = grades_info.apply(lambda x: x['has_middle'])
    df['has_high_school'] = grades_info.apply(lambda x: x['has_high'])
    df['has_special_school'] = grades_info.apply(lambda x: x['has_special'])

    # Создаем комбинированный признак
    df['school_levels_count'] = (
            df['has_elementary_school'].astype(int) +
            df['has_middle_school'].astype(int) +
            df['has_high_school'].astype(int)
    )

    def extract_school_names_info(schools_list):
        """Извлекаем название школ"""

        if not isinstance(schools_list, list):
            return {'names': [], 'types': []}

        names = []
        school_types = []

        for school in schools_list:
            if isinstance(school, dict):
                name_list = school.get('name', [])
                if name_list and isinstance(name_list, list) and len(name_list) > 0:
                    name = str(name_list[0]).upper()
                    names.append(name)

                    # Определяю тип школы по названию
                    if any(word in name for word in ['ELEMENTARY', 'PRIMARY']):
                        school_types.append('elementary')
                    elif any(word in name for word in ['MIDDLE', 'JUNIOR']):
                        school_types.append('middle')
                    elif any(word in name for word in ['HIGH', 'SENIOR']):
                        school_types.append('high')
                    elif any(word in name for word in ['ACADEMY', 'CHARTER']):
                        school_types.append('charter')
                    elif any(word in name for word in ['INSTITUTE', 'TECH', 'VOC']):
                        school_types.append('vocational')
                    elif any(word in name for word in ['MAGNET', 'MONTESSORI']):
                        school_types.append('special')
                    else:
                        school_types.append('other')

        return {'names': names, 'types': school_types}

    # Применяю
    schools_info = df['schools_parsed'].apply(extract_school_names_info)

    # Создаю признаки по типам школ
    def count_school_type(info, school_type):
        return sum(1 for t in info['types'] if t == school_type)

    df['num_elementary_schools'] = schools_info.apply(lambda x: count_school_type(x, 'elementary'))
    df['num_middle_schools'] = schools_info.apply(lambda x: count_school_type(x, 'middle'))
    df['num_high_schools'] = schools_info.apply(lambda x: count_school_type(x, 'high'))
    df['num_charter_schools'] = schools_info.apply(lambda x: count_school_type(x, 'charter'))

    def calculate_school_district_score(row):
        """Расчет оценок"""
        score = 0

        # Базовые баллы за количество школ
        num_schools = len(row.get('school_ratings_list', []))
        if num_schools >= 5:
            score += 3
        elif num_schools >= 3:
            score += 2
        elif num_schools >= 1:
            score += 1

        # Качество школ (средний рейтинг)
        avg_rating = row.get('avg_school_rating', 0)
        if avg_rating >= 8:
            score += 3
        elif avg_rating >= 6:
            score += 2
        elif avg_rating >= 4:
            score += 1

        # Близость школ
        min_dist = row.get('min_school_distance_mi', 100)
        if min_dist <= 0.5:
            score += 3
        elif min_dist <= 1:
            score += 2
        elif min_dist <= 2:
            score += 1

        # Разнообразие уровней
        levels_count = row.get('school_levels_count', 0)
        score += min(levels_count, 3)  # максимум 3 балла

        # Наличие хороших школ (рейтинг >= 7)
        good_schools = row.get('num_good_schools', 0)
        if good_schools >= 3:
            score += 3
        elif good_schools >= 2:
            score += 2
        elif good_schools >= 1:
            score += 1

        return min(score, 10)  # Ограничиваю максимум 10

    df['school_district_score'] = df.apply(calculate_school_district_score, axis=1)

    def categorize_school_score(score):
        """Категоризируем школы по очкам"""
        if score >= 8:
            return 'excellent'
        elif score >= 6:
            return 'good'
        elif score >= 4:
            return 'average'
        elif score >= 2:
            return 'poor'
        else:
            return 'very_poor'

    df['school_district_cat'] = df['school_district_score'].apply(categorize_school_score)

    # Проверяю наличие школ определенных типов в названиях
    def check_school_keywords(info, keywords):
        """Признак престижных школ"""
        names = ' '.join(info['names'])
        return any(keyword.upper() in names for keyword in keywords)

    # Популярные "престижные" ключевые слова
    prestige_keywords = ['ACADEMY', 'MAGNET', 'CHARTER', 'PREP', 'PREPARATORY']
    df['has_prestige_school'] = schools_info.apply(
        lambda x: check_school_keywords(x, prestige_keywords)
    )

    # Школы с именами известных людей (часто показатель качества)
    famous_names_keywords = ['WASHINGTON', 'LINCOLN', 'JEFFERSON', 'ROOSEVELT', 'KENNEDY']
    df['has_famous_name_school'] = schools_info.apply(
        lambda x: check_school_keywords(x, famous_names_keywords)
    )

    def clean_sqft(value):
        """Очищаю площадь"""

        if pd.isna(value):
            return 0

        # привожу к строке
        s = str(value).lower().strip()

        s = s.replace("total interior livable area:", "")

        s = s.replace("sqft", "").replace('"', "").replace("'", "")

        s = s.replace(",", "")

        s = re.sub(r"[^\d.]", "", s)

        try:
            return float(s)
        except:
            return 0

    def categorize_sqft(x):
        """Категоризирую площадь"""

        if pd.isna(x):
            return "missing"
        elif x < 5000:
            return "small"
        elif x <= 10000:
            return "medium"
        else:
            return "large"

    df["sqft_clean"] = df["sqft"].apply(clean_sqft)
    df["sqft_category"] = df["sqft_clean"].apply(categorize_sqft)

    def clean_beds(value):
        """Очищаю признак спален"""

        if not isinstance(value, str):
            return 0

        val = value.lower()

        # Если встречаются нечисловые единицы — сразу 0
        if re.search(r"sqft|acres?|bath", val):
            return 0

        # '3 or more' → 3
        if '3 or more' in val:
            return 3

        # Ищу цифру в начале строки
        match = re.search(r'\d+', val)
        if match:
            return int(match.group())

        return 0

    df['beds_clean'] = df['beds'].apply(clean_beds)

    def fireplace_value(value):
        v = str(value).lower()

        # 1. Количество
        count = 0
        if '1' in v or 'one' in v:
            count = 1
        if '2' in v or 'two' in v:
            count = 2
        if '3' in v or 'three' in v or '4' in v or 'four' in v:
            count = 3

        # 2. Тип
        fp_type = 'unknown'
        if 'wood' in v:
            fp_type = 'wood'
        elif 'gas' in v:
            fp_type = 'gas'
        elif 'electric' in v:
            fp_type = 'electric'
        elif 'decorative' in v:
            fp_type = 'decorative'
        elif 'pellet' in v:
            fp_type = 'pellet'

        # 3. Расположение
        room_count = 0
        rooms = ['living', 'family', 'great', 'master', 'bedroom', 'den', 'basement', 'kitchen', 'dining']

        for room in rooms:
            if room in v:
                room_count += 1

        location = 'unknown'
        if room_count == 0:
            location = 'unknown'
        elif room_count == 1:
            if 'living' in v:
                location = 'living'
            elif 'family' in v:
                location = 'family'
            elif 'great' in v:
                location = 'great'
            elif 'master' in v:
                location = 'master'
            elif 'bedroom' in v:
                location = 'bedroom'
            elif 'den' in v:
                location = 'den'
            elif 'basement' in v:
                location = 'basement'
            elif 'kitchen' in v:
                location = 'kitchen'
            elif 'dining' in v:
                location = 'dining'
        else:
            location = 'multiple'

        # 4. Есть ли камин?
        has_fp = 0  # предполагаем что нет

        # Проверяю отсутствия
        if count > 0 or room_count > 0 or fp_type != 'unknown':
            has_fp = 1

        if has_fp == 0:
            return 0, 0, 'unknown', 'unknown'

        return has_fp, count, fp_type, location

    # Создаю новый DataFrame из результатов
    fireplace_features = df['fireplace'].apply(lambda x: pd.Series(fireplace_value(x)))

    # Объединяю с исходными данными
    df = pd.concat([df, fireplace_features], axis=1)

    # Переименовываю
    df = df.rename(columns={
        0: 'has_fireplace',
        1: 'fireplace_count',
        2: 'fireplace_type',
        3: 'fireplace_location'
    })

    # ТОП-50 городов США по населению
    top_50_cities = [
        'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
        'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
        'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
        'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Washington',
        'Boston', 'El Paso', 'Nashville', 'Detroit', 'Oklahoma City',
        'Portland', 'Las Vegas', 'Memphis', 'Louisville', 'Baltimore',
        'Milwaukee', 'Albuquerque', 'Tucson', 'Fresno', 'Sacramento',
        'Kansas City', 'Long Beach', 'Mesa', 'Atlanta', 'Colorado Springs',
        'Virginia Beach', 'Raleigh', 'Omaha', 'Miami', 'Oakland',
        'Minneapolis', 'Tulsa', 'Arlington', 'New Orleans', 'Wichita'
    ]

    # Создаю признак: 1 если город в топ-50, 0 если нет
    df['city_cat'] = df['city'].apply(
        lambda x: 1 if str(x).title() in top_50_cities else 0
    )

    def city_size_tier(city):
        """Разбиваю города по категориям"""

        city_name = str(city).title() if pd.notna(city) else ""

        if city_name in ['New York', 'Los Angeles', 'Chicago']:
            return 'tier_1 - Megacity'
        elif city_name in ['Houston', 'Phoenix', 'Philadelphia', 'San Antonio',
                           'San Diego', 'Dallas', 'San Jose']:
            return 'tier_2 - Major'
        elif city_name in top_50_cities:
            return 'tier_3 - Large'
        elif city_name:
            return 'tier_4 - Other'
        else:
            return 'unknown'

    df["city"] = df["city"].fillna("MISSING")

    df['city_tier'] = df['city'].apply(city_size_tier)

    def clean_stories(value):
        """Очищаю признак этажность"""

        if pd.isna(value) or str(value).strip() in ['', 'MISSING']:
            return 1

        value_str = str(value).strip()

        # Словарь для текстовых значений
        text_mapping = {
            # Один этаж
            'one': 1, 'one story': 1, 'one level': 1, 'ranch': 1,
            'ranch/1 story': 1, '1 story/ranch': 1, 'one story/ranch': 1,
            '1 story': 1, '1 level': 1,

            # Два этажа
            'two': 2, 'two story': 2, '2 story': 2, '2 stories': 2,
            'two stories': 2, 'two story or more': 2, '2 story or more': 2,
            'two story/basement': 2, '2 story/basement': 2,
            '2 or more stories': 2, 'townhouse': 2, 'condominium': 2,

            # Три и более
            'three or more': 3, 'three': 3, '3 story': 3, '3+': 3,
            'tri-level': 3,

            # Дробные
            'one and one half': 1.5, '1.5 story': 1.5, '1.5 level': 1.5,
            '2.5 story': 2.5,

            # Нулевые значения
            'lot': 0, 'acreage': 0,

            # Типы зданий
            'mid-rise': 5, 'high-rise': 10,
            'multi/split': 2, 'split level': 2, 'bi-level': 2,
        }

        # Проверяю текстовые значений
        lower_val = value_str.lower()
        if lower_val in text_mapping:
            return text_mapping[lower_val]

        # Извлекаю числа из строк вида "2 Level, Site Built"
        match = re.search(r'(\d+(?:\.\d+)?)', value_str)
        if match:
            return float(match.group(1))

        # Попытка преобразовать в число
        try:
            num_val = float(value_str)
            return int(num_val) if num_val.is_integer() else num_val
        except:
            # Если не удалось распознать - ставим 1
            return 1

    df['stories_clean'] = df['stories'].apply(clean_stories)

    # Перечень признаков которые пойдут в модель
    cols_to_use = ['status_cat', 'city_tier', 'street_cat', 'sqft_category', 'propertyType_cat',
                   'lotsize_cat', 'heating_cat', 'cooling_cat', 'parking_cat',
                   'stories_clean', 'pool', 'baths_clean', 'sqft_clean', 'beds_clean',
                   'fireplace_type', 'has_fireplace', 'fireplace_count', 'fireplace_location',
                   'Year built', 'Remodeled year',
                   'lotsize_clean',
                   'avg_school_rating', 'max_school_rating', 'num_good_schools',
                   'min_school_distance_mi', 'avg_school_distance_mi',
                   'schools_within_1mi', 'has_elementary_school',
                   'has_middle_school', 'has_high_school', 'has_special_school',
                   'school_levels_count', 'num_elementary_schools', 'num_middle_schools', 'num_high_schools',
                   'num_charter_schools', 'school_district_score', 'school_district_cat', 'has_prestige_school',
                   'has_famous_name_school']

    # Создаю новый датафрейм
    data_to_use = df[cols_to_use].copy()

    # Були
    columns_bool = ['pool', 'has_elementary_school', 'has_middle_school', 'has_high_school',
                    'has_special_school', 'has_prestige_school', 'has_famous_name_school']

    for col in columns_bool:
        if col in data_to_use.columns:
            data_to_use[col] = data_to_use[col].fillna(0).astype(int)

    # Пост-обработка строковых колонок
    def clean_string_columns(df):
        """
        Финальная очистка всех строковых колонок
        """
        # Нахожу все строковые колонки
        string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

        for col in string_cols:
            # Заполняю пропуски
            df[col] = df[col].fillna('unknown')

            # Преобразую в строку
            df[col] = df[col].astype(str)

            # Очищаю
            df[col] = df[col].str.strip().str.lower()
            df[col] = df[col].str.replace(',', '', regex=False)

            # Заменяю пустые и 'nan' строки
            df[col] = df[col].replace(['', 'nan', 'none', 'null'], 'unknown')

        return df

    # Применяю финальную очистку
    data_model = clean_string_columns(data_to_use)

    return  data_model
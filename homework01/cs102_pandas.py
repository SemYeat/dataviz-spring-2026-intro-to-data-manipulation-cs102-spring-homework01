import pandas as pd
import re
from typing import Tuple


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir = data.loc[data["факультет"].eq("факультет систем управления и робототехники")].copy()
    return len(fsuir), fsuir["группа"].nunique(), fsuir


# Задача 2
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    """
    Проверяет наличие однофамильцев на ФСУиР, их количество, распределение по курсам
    и определяет группу с наибольшим числом однофамильцев.
    Возвращает:
     - логическое значение (наличие однофамильцев)
     - общее количество однофамильцев
     - серию с числом однофамильцев по курсам
     - группу с максимальным числом однофамильцев
    """
    surnames = df["фио"].str.split().str[0]
    # keep=False отмечает всех носителей повторяющейся фамилии.
    homonyms = df.loc[surnames.duplicated(keep=False)]
    per_course = homonyms.groupby("курс").size()
    groups = homonyms.groupby("группа").size()
    max_group = groups.idxmax() if not groups.empty else ""
    return not homonyms.empty, len(homonyms), per_course, max_group
    

# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if not isinstance(patronym, str):
        return "unknown"
    patronym = patronym.strip().lower()
    if re.search(r"(?:овна|евна|ична|инична)$", patronym):
        return "female"
    if re.search(r"(?:ович|евич|ич)$", patronym):
        return "male"
    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу 
    """
    patronyms = df["фио"].str.split().str[2].fillna("")
    # Отсутствие третьего слова и нераспознанное отчество — разные случаи.
    without_patronym = int(patronyms.eq("").sum())
    genders = patronyms.loc[patronyms.ne("")].map(gender_identification)
    return without_patronym, genders.value_counts()


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    counts = data.groupby("факультет").size().rename("количество_студентов")
    max_faculty, min_faculty = counts.idxmax(), counts.idxmin()
    return (counts.reset_index(),
            (max_faculty, int(counts.loc[max_faculty])),
            (min_faculty, int(counts.loc[min_faculty])))


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    # Сначала число студентов в каждой существующей паре факультет–курс.
    counts = data.groupby(["курс", "факультет"]).size()
    by_course = counts.groupby(level="курс")
    return by_course.mean().round(1), by_course.median()


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, str, float]:
    """
    Определяет самое популярное имя, группу с наибольшим количеством студентов с этим именем,
    факультет, курс и долю таких студентов в общем числе.
    Возвращает результат в следующем порядке:
     1. самое частое имя
     2. группа
     3. факультет
     4. курс
     5. доля
    """
    names = data["фио"].str.split().str[1]
    popular_name = names.value_counts().idxmax()
    selected = data.loc[names.eq(popular_name)]
    group = selected.groupby("группа").size().idxmax()
    row = selected.loc[selected["группа"].eq(group)].iloc[0]
    ratio = round(len(selected) / len(data), 2)
    return popular_name, group, row["факультет"], row["курс"], ratio


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    names = data["фио"].str.split().str[1]
    counts = names.map(names.value_counts())
    mask = counts.eq(1) & names.str.startswith("П", na=False)
    return data.loc[mask, ["фио", "факультет", "курс"]].copy()


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    third_course = data.loc[data["курс"].eq("3-й")].copy()
    faculty = third_course.groupby("факультет")["средний_балл"].mean().idxmax()
    selected = third_course.loc[third_course["факультет"].eq(faculty)].copy()
    patronyms = selected["фио"].str.split().str[2].fillna("")
    selected["пол"] = patronyms.map(gender_identification)
    known = selected.loc[selected["пол"].isin(["female", "male"])]
    averages = known.groupby("пол")["средний_балл"].mean()
    gender = averages.idxmax()
    return faculty, gender, int(round(averages.loc[gender]))


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    ordered = data.sort_values("ису").reset_index(drop=True)
    # Новый блок начинается при разрыве последовательности номеров.
    blocks = ordered["ису"].diff().ne(1).cumsum()
    lengths = ordered.groupby(blocks)["ису"].transform("size")
    eligible = ordered.loc[lengths.ge(5)]
    # ИСУ оставлен для проверки последовательности приложенным тестом.
    return eligible.head(5)[["фио", "факультет", "курс", "группа", "ису"]].copy()


if __name__ == "__main__":
    from pathlib import Path

    data = pd.read_csv(Path(__file__).with_name("isu_fake_data.csv"))
    
    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, Групп: {num_groups}")
    
    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = find_homonymous_students(fsuir)
    print(f"Есть однофамильцы: {has_homonyms}, Всего: {total_homonyms}, Группа с максимумом: {max_homonym_group}")
    print(f"На каждом курсе: {homonyms_per_course}")
    
    # Задача 3
    students_without_patronym, gender_counts = analyze_patronyms(fsuir)
    print(f"Студентов без отчества: {students_without_patronym}")
    print("Распределение по полу:", gender_counts)
    
    # Задача 4
    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print(faculty_counts.to_string(index=False))
    print(f"Факультет с наибольшим числом студентов: {max_faculty}")
    print(f"Факультет с наименьшим числом студентов: {min_faculty}")
    
    # Задача 5
    mean_students, median_students = course_statistics(data)
    print("Среднее число студентов на курсах:", mean_students)
    print("Медианное число студентов на курсах:", median_students)
    
    # Задача 6
    popular_name, name_group, faculty, course, name_ratio = most_popular_name(data)
    print(f"Самое популярное имя: {popular_name}, Группа: {name_group}, Факультет: {faculty}, Курс: {course}")
    print(f"Доля студентов с этим именем: {name_ratio}")
    
    # Задача 7
    result_7 = find_students_with_name_starting_P(data)
    print("Студенты с именем, начинающимся на П и встречающимся ровно один раз:")
    print(result_7.to_string(index=False))
    
    # Задача 8
    fac, best_gender, best_grade = highest_avg_grade_faculty(data)
    print(f"Факультет с высоким средним баллом 3-го курса: {fac}")
    print(f"Пол с наивысшим средним баллом: {best_gender}, Средний балл: {best_grade}")
    
    # Задача 9
    result_9 = find_consecutive_students(data)
    print("Первые 5 студентов с подряд идущими табельными номерами:")
    print(result_9[["фио", "факультет", "курс", "группа"]].to_string(index=False))

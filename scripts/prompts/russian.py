import random

from scripts.prompts.utils import MedicalRecord


def prompt_no_patent_info(symptoms, diagnose, marital, smoking, gender, conj):
    return f"""
    Напиши ответ как будто являешься профессиональным врачом, который записывает анамнез со слов пациента. Напиши хорошо структурированный и детальный анамнез.
    Пациент жалуется на {symptoms}.
    Реалистично придумай все недостающие детали, характеристики пациента и сопутствующие симптомы для анамнеза. """


def prompt_with_patient_info(symptoms, diagnose, marital, smoking, gender, conj):
    return f"""
Напиши текст анамнеза, составленного врачом по итогам приема пациента.
Пациент - {marital} {smoking} {gender}, {conj} жалуется на {symptoms}.
Придумай все недостающие детали, характеристики пациента и сопутствующие симптомы для анамнеза. """


def short_promt(symptoms, diagnose, marital, smoking, gender, conj):
    return f"""
Представь, что являешься врачом. Напиши краткий анамнез по итогам приема пациента, который жалуется на {symptoms}.
 Дополни все недостающие детали реалистичными данными. """


def prompt_diagnostics(symptoms, diagnose, marital, smoking, gender, conj):
    return f"""
    Напиши ответ как будто являешься профессиональным врачом, который записывает анамнез со слов пациента. Напиши хорошо структурированный и детальный анамнез.
    Пациент жалуется на {symptoms}.
    Реалистично придумай все недостающие детали и результаты исследований, которые могли быть назначены в такой ситуации. """


PROMPTS = [
    prompt_no_patent_info,
    prompt_with_patient_info,
    short_promt,
    prompt_diagnostics,
]


def get_russian_details(gender, marital_state, smoking):

    if gender == "male":
        marital = "женатый" if marital_state else "неженатый"
        smoking = "курящий" if smoking else "некурящий"
        gender_ru = "мужчина"
        conj = "который"
    else:
        marital = "замужняя" if marital_state else "незамужняя"
        smoking = "курящая" if smoking else "некурящая"
        gender_ru = "женщина"
        conj = "которая"
        # Известно, что пациент - {marital} {smoking} {gender_ru}.
    return conj, gender_ru, marital, smoking


def get_desase_tamplate(sample: dict) -> MedicalRecord:
    desease_code = sample["disease"][0]["idc_10"]
    symptoms = sample["symptoms"]
    # age = data["age"]
    gender = sample["gender"]
    marital_state = sample["family_state"]
    smoking = sample["smoking"]

    desaese_name = str(sample["disease"][0]["name_ru"]).lower()

    conj, gender_ru, marital, smoking = get_russian_details(
        gender, marital_state, smoking
    )

    return MedicalRecord(
        desease_code=desease_code,
        symptoms=symptoms,
        gender=gender,
        marital_state=marital_state,
        smoking=bool(smoking),
        desease_name=desaese_name,
        prompt=random.choice(PROMPTS)(
            ", ".join(symptoms).lower(), desaese_name, marital, smoking, gender_ru, conj
        ),
    )

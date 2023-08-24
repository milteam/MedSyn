import random

from scripts.prompts.utils import MedicalRecord


def get_desase_tamplate(sample: dict) -> MedicalRecord:
    desaese_name = sample["disease"][0]["Name"]
    uid = sample["UID"]
    desease_code = sample["disease"][0]["idc_10"]
    symptoms = sample["symptoms"]
    gender = sample["gender"]
    marital_state = sample["family_state"]
    smoking = sample["smoking"]

    if len(symptoms) > 3:
        symptoms = random.sample(symptoms, random.randint(1, 4))

    return MedicalRecord(
        uid=uid,
        desease_code=desease_code,
        symptoms=symptoms,
        gender=gender,
        marital_state=marital_state,
        smoking=bool(smoking),
        desease_name=desaese_name,
        prompt=f"""
                Act like a professional doctor who listened to the patient. Write a well-structured and extremely detailed medical anamnesis that includes all the required sections.        
              In fact, the patient has {desaese_name}, but the doctor does not know about it.
              The patient complains of {", ".join(symptoms)}.
              """,
    )

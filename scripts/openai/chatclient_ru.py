import os

import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-4-0314",
    messages=[
        # {"role": "system",
        #  "content": "Act like a professional doctor who listened to the patient. Write a well-structured and extremely detailed medical anamnesis that includes all the required sections."},
        {"role": "user", "content":
            '''
            Напиши ответ как будто являешься профессиональным врачом, который записывает анамнез со слов пациента. Напиши хорошо структурированный и детальный отчет.
Представь, что у пациента есть слеующие хронические заболевания: гастрит, тонзилит.
В действительности диагноз пациента - обострение хронического гастратита. Но на момент написания анамнеза это еще неизвестно.
Пациент жалуется на рвоту и боль в животе.'''},
    ]
)
print(response['choices'][0]['message']['content'])
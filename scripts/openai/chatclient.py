import os

import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")
'''
PROMPT:
“Act like a professional doctor who listened to the patient. Write a well-structured and extremely detailed medical anamnesis that includes all the required sections.
The patient has chronic diseases: gastritis, tonsillitis.
In fact, the patient has an exacerbation of chronic gastritis, but the doctor does not know about it.
The patient complains of stomach pain and vomiting.
Mark up the text: highlight all the symptoms and medical terms in square brackets, write down the type of the word”
'''
# response = openai.Completion.create(model="text-davinci-003", prompt="Say this is a test", temperature=0, max_tokens=7)
response = openai.ChatCompletion.create(
    model="gpt-4-0314",
    messages=[
        # {"role": "system",
        #  "content": "Act like a professional doctor who listened to the patient. Write a well-structured and extremely detailed medical anamnesis that includes all the required sections."},
        {"role": "user", "content":
            '''
            Act like a professional doctor who listened to the patient. Write a well-structured and extremely detailed medical anamnesis that includes all the required sections.
            The patient has chronic diseases: gastritis, tonsillitis.
  In fact, the patient has an exacerbation of chronic gastritis, but the doctor does not know about it.
  The patient complains of stomach pain and vomiting.
  Mark up the text: highlight all the symptoms and medical terms in square brackets, write down the type of the word.'''},
    ]
)
print(response['choices'][0]['message']['content'])

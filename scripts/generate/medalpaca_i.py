from model import load_model

inferer = load_model()

print(inferer('''
Write a well-structured  anamnesis that includes all the required sections.
In fact, the patient has an exacerbation of chronic gastritis, but the doctor does not know about it.
The patient complains of stomach pain and vomiting.
'''))

import torch

from model import load_model
assert torch.cuda.is_available(), "No cuda device detected"

inferer = load_model()

print(inferer('''
Act like a professional doctor who listened to the patient. Write a well-structured and extremely detailed medical anamnesis that includes all the required sections.
The patient has chronic diseases: gastritis, tonsillitis.
In fact, the patient has an exacerbation of chronic gastritis, but the doctor does not know about it.
The patient complains of stomach pain and vomiting.
Mark up the text: highlight all the symptoms and medical terms in square brackets, write down the type of the word

'''))

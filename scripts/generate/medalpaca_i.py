import torch

from model import load_model
assert torch.cuda.is_available(), "No cuda device detected"

inferer = load_model()

print(inferer('''
Write a well-structured  anamnesis that includes all the required sections.
In fact, the patient has an exacerbation of chronic gastritis, but the doctor does not know about it.
The patient complains of stomach pain and vomiting.
''',
              instruction="Act as a professional doctor"))

from collections import namedtuple

DiseaseInfo = namedtuple('DiseaseInfo', ['name', 'symptoms', 'preconditions', 'code'])

graph = [
    DiseaseInfo(name='вирус Коксаки', code='мкб-100', symptoms=['диарея', 'сыпь на руках'], preconditions=[])
]


def sample_disease():
    return graph[0]

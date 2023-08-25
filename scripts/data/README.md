## Данные для обучения и бенчмарков:

Конвертация реальных анамнезов в формат RuMedTop3:
```
python3 scripts/data/samples_to_top3.py
```

Подготовка данных для IFT:
```
python3 scripts/data/prepare_wikimed.py  # мед. данные
python3 scripts/data/prepare_sber_data.py  # реальные анамнезы
python3 scripts/data/prepare_llama.py  # сгенерированные анамнезы
```

Подготовка данных для RuMedTop3 (апсемплинг и даунсемплинг):
```
python3 scripts/data/prepare_top3.py
pytohn3 scripts/data/prepare_anam_top3.py  # только сгенерированные данные
```
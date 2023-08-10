## self_instruct finetuning

### Структура папок

```
/self_instruct
├── models                     # загруженные и обученные локальные чекпоинты
│   ├── decapoda-llama-7b 
│   └── ...
├── output                     # здесь лежат результаты генерации
│   └── ...
├── templates                  # шаблоны инструкций для alpaca и чата для rugpt
│   ├── ru_alpaca.json
│   └── ...
├── utils
│   └── ...
├── adapter_finetune.py / .sh  # скрипт для файнтюна llama adapter
                               # не использую пока
├── Dockerfile
├── download_models.py / .sh   # скрипт для загрузки чекпоинтов модели. 
                               # возможно, нужно будет использовать, 
                               # чтобы загрузить модели дополнительно в кэш
                               
├── finetune.py / .sh          # скрипт для тюна всех моделей
                               # пример запуска есть в файле `finetune.sh`
                               
├── finetune_old.py / .sh      # старый скрипт для файнтюна, не используется

├── generate.py / .sh          # генерация с gui, можно менять некоторые
                               # параметры generation_config
                               
├── hf_generation.py / .sh     # скрипт из официального туториала, в принципе можно его использовать

├── infer_alpaca.py / .sh      # скрипты для проверки генерации на 3х примерах 
├── infer_llama.py / .sh
├── infer_saiga.py / .sh

├── requirements.txt
└── test.jsonl                 # тестовые данные, которые используются скриптами infer_*.sh
```

### Путь к загруженным чекпоинтам
```
/home/kuzkina/MedTexts/alpaca-lora/models/
```

Поскольку `llama_v2` можно загружать из huggingface только с токеном, 
я скачала веса 1 раз и они огромной папкой лежат в `models/`. Я пока не придумала,
как эффективно загружать эту модель в докере. Пока приходится целиком её копировать, 
а это немного долго, в отличие от использования кешированных моделей.

### Путь к некоторым результатам генерации
```
/home/kuzkina/MedTexts/alpaca-lora/output/
```

### Основная команда запуска файнтюнинга
```
sh finetune.sh
```

Возможно, запуск на локальных моделях не будет работать, 
для этого можно использовать чекпоинты huggingface 
или попробовать загрузить чекпоинты ещё раз с помощью `download_models.sh`.

Код для тюнинга и генерации в основном взят отсюда [IlyaGusev/rulm](https://github.com/IlyaGusev/rulm/blob/master/self_instruct)
и отсюда [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora).

### Команда для генерации
```
sh infer_alpaca.sh
```

### Используемые модели

1) LLaMA2 (Saiga2)

    Веса LLaMA2 локальные: `models/meta-llama-v2-7b`

    Веса Saiga2 локальные: `models/ru_saiga-v2_7b`

    Параметры запуска:
    ```
      python3.10 finetune.py \
        --base_model='models/meta-llama-v2-7b' \
        --resume_from_checkpoint="models/ru_saiga-v2_7b" \
        --only_target_loss=True \
        --data_path='data/alpaca_med_data_10k.json' \
        --template_path="templates/saiga_v2.json" \
        --model_type="causal" \
        --mode="chat" \
        --num_epochs=3 \
        --max_tokens_count=2000 \
        --learning_rate=1e-5 \
        --group_by_length \
        --output_dir=models/"$OUTPUT_DIR" \
        --lora_target_modules='[q_proj,v_proj,k_proj,o_proj]' \
        --lora_r=16 \
        --lora_alpha=16 \
        --micro_batch_size=8 \
        --warmup_steps=10 \
        --val_set_size=0
    ```
   Подробнее об обучении можно посмотреть [здесь](https://huggingface.co/IlyaGusev/saiga2_7b_lora) и [здесь](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/saiga2_7b.json) 

2) RuGPT3.5 (gigasaiga)
    
    Веса rugpt локальные: `models/rugpt-13b`

    Веса гигасайги локальные: `models/ru_gigasaiga_lora`

    Подробнее об обучении можно посмотреть [здесь](https://huggingface.co/IlyaGusev/gigasaiga_lora) и [здесь](https://github.com/IlyaGusev/rulm/blob/master/self_instruct/configs/gigasaiga_13b.json)

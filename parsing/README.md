На примере [WikiMed](http://wikimed.pro/index.php?title=%D0%92%D0%B8%D0%BA%D0%B8%D0%BC%D0%B5%D0%B4), аналогино можно выполнить для [MayoClinic](https://www.mayoclinic.org/).
```
mkdir ./data/wikimed
cd ./parsing/wikimed

# Parse disease/meds pages and their links:
scrapy crawl all_data -O ../../data/wikimed/all_data.json

# Parse diseases data to csv table:
python3 ./parsing/wikimed/parse_diseases.py

# Parse meds pages and their links:
scrapy crawl meds -O ../../data/wikimed/meds.json

# Parse meds data to csv table:
python3 ./parsing/wikimed/parse_meds.py
```


scrapy crawl symptoms -O ../../data/msd/symptoms.json

scrapy crawl diseases -O ../../data/msd/diseases.json


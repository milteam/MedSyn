# Medical Text Generation

## Samples Generation:
To generate samples:
1. Define config `scripts/sampler/cfg.yaml`
2. Run 
```
python3 ./scripts/sampler/generate.py
```

## Data Description:
* `data/codes/top_ICD10_codes.txt` - top frequent diseases from Sber data.
* `data/codes/icd_gender_split.json` - split of diseases codes basesd on gender, extracted from [ICD10Volume2_en_2019](https://icd.who.int/browse10/Content/statichtml/ICD10Volume2_en_2019.pdf).
* `data/demographical/EdgeGenderRU_2022.csv` - edge and gender data for Russian Federation for 2022 from [rosstat](https://rosstat.gov.ru/compendium/document/13284). Manually cleaned version of page 1.1. for 2022.
* `data/demographical/FamilyStatusRU` - family status data from [rosstat](https://rosstat.gov.ru/storage/mediabank/demo33_2021.xls).
* `data/demographical/EthnicGroups.csv` - top 79 ethnical groups (with ration >= 0.01%) from [wikipedia](https://ru.wikipedia.org/wiki/%D0%9D%D0%B0%D1%86%D0%B8%D0%BE%D0%BD%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9_%D1%81%D0%BE%D1%81%D1%82%D0%B0%D0%B2_%D0%A0%D0%BE%D1%81%D1%81%D0%B8%D0%B8).

* `data/mesh2icd10.txt` - [source](https://github.com/tonifuc3m/Mesh2ICD10/tree/master)

* `data/MESH_ICD10_Mapping.csv` - [source](https://github.com/kush02/Automated-ICD10-Codes-Assignment/tree/master)

* `data/symptoms-DO.tsv` - [source](https://think-lab.github.io/d/52/)


## WikiMed parsing:
[WikiMed](http://wikimed.pro/index.php?title=%D0%92%D0%B8%D0%BA%D0%B8%D0%BC%D0%B5%D0%B4)
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
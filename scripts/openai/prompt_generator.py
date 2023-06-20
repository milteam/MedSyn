import json

from graph import DiseaseInfo



def generate_prompt(info: DiseaseInfo):
    return [{"role": "user", "content": f'''
            Напиши ответ как будто являешься профессиональным врачом, который записывает анамнез со слов пациента. Напиши хорошо структурированный и детальный отчет.
Представь, что у пациента есть слеующие хронические заболевания: {", ".join(info.preconditions)}.
В действительности диагноз пациента - {info.name}. Но на момент написания анамнеза это еще неизвестно.
Пациент жалуется на {", ".join(info.symptoms)}. Реалистично придумай все недостающие данные для отчета.'''},
            ]


@click.command()
@click.option('--taskid', '-t', help='Task id', multiple=True, required=True, type=int)
def main():
    pass


if __name__ == '__main__':
    main()
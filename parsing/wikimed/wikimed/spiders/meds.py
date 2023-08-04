import scrapy

IGNORE_LIST = [
    "Представиться системе",
    "Создать учётную запись",
]


class WikiMedSpider(scrapy.Spider):
    name = "meds"
    allowed_domains = ["wikimed.pro"]
    start_urls = [
        "http://wikimed.pro/index.php?title=%D0%94%D0%B5%D0%B9%D1%81%D1%82%D0%B2%D1%83%D1%8E%D1%89%D0%B8%D0%B5_%D0%B2%D0%B5%D1%89%D0%B5%D1%81%D1%82%D0%B2%D0%B0",
    ]

    def parse(self, response):
        links = response.xpath("/html/body/div/div/div/ul/li/a")
        for link in links:
            next_page = link.xpath(".//@href").get()
            if next_page is not None:
                next_page = response.urljoin(next_page)
                name = response.css("h1::text").get()
                if name not in IGNORE_LIST:
                    yield response.follow(url=next_page, callback=self.parse_link)

    def parse_link(self, response):
        name = response.css("h1::text").get()
        if name not in IGNORE_LIST:
            yield {
                "name": name,
                "link": response.url,
            }

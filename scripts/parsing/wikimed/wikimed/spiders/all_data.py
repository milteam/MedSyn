import scrapy

IGNORE_LIST = [
    "Представиться системе",
    "Создать учётную запись",
]


class WikiMedSpider(scrapy.Spider):
    name = "all_data"
    allowed_domains = ["wikimed.pro"]
    start_urls = [
        "http://wikimed.pro/",
        "http://wikimed.pro/index.php?title=%D0%92%D0%B8%D0%BA%D0%B8%D0%BC%D0%B5%D0%B4",
    ]

    def parse(self, response):
        links = response.xpath("/html/body/div/div/div/ul/li/a")
        for link in links:
            next_page = link.xpath(".//@href").get()
            if next_page is not None:
                next_page = response.urljoin(next_page)
                name = response.css("h1::text").get()
                if name not in IGNORE_LIST:
                    yield response.follow(url=next_page, callback=self.parse)

        name = response.css("h1::text").get()
        if name not in IGNORE_LIST:
            yield {
                "name": name,
                "secondary_name": response.xpath(
                    "/html/body/div[3]/div[2]/div[4]/p[1]/b/text()"
                ).get(),
                "link": response.url,
            }

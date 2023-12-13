import scrapy

IGNORE_LIST = [
    # "Disclaimer",
    # "Ресурсы",
    # "Permissions",
    # "Overview of the MSD Manuals",
    # "The Trusted Provider of Medical Information since 1899",
    # "О Справочниках MSD",
    # ' Справочник MSD ,',
    # "Симптомы"
]

class MsdSpider(scrapy.Spider):
    name = "symptoms"
    allowed_domains = ["msdmanuals.com"]
    start_urls = [
        "https://www.msdmanuals.com/",
        "https://www.msdmanuals.com/ru/%D0%B4%D0%BE%D0%BC%D0%B0/symptoms",
    ]

    def parse(self, response):
        links = response.xpath('//li/a/@href')
        for link in links:
            next_page = link.get()
            if next_page is not None:
                next_page = response.urljoin(next_page)
                name = response.css("h1::text").get()
                if name not in IGNORE_LIST:
                    yield response.follow(url=next_page, callback=self.parse_link)

    def parse_link(self, response):
        name = response.css("h1::text").get()
        name = name.replace("\r\n", "").strip()
        if name not in IGNORE_LIST:
            yield {
                "name": name,
                "link": response.url,
            }

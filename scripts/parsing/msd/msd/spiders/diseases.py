import scrapy


class MsdSpider(scrapy.Spider):
    name = "diseases"
    allowed_domains = ["msdmanuals.com"]
    start_urls = [
        "https://www.msdmanuals.com/",
        "https://www.msdmanuals.com/ru/%D0%BF%D1%80%D0%BE%D1%84%D0%B5%D1%81%D1%81%D0%B8%D0%BE%D0%BD%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9/health-topics",
    ]

    def parse(self, response):
        links = response.xpath("//a[contains(@class, 'section__item')]/@href")
        for link in links:
            next_page = link.get()
            if next_page is not None:
                next_page = response.urljoin(next_page)
                name = response.css("h1::text").get()
                yield response.follow(url=next_page, callback=self.parse_link)

    def parse_link(self, response):
        name = response.css("h1::text").get()
        yield {
            "name": name,
            "link": response.url,
        }

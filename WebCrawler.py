from bs4 import BeautifulSoup
from time import sleep
import urllib.request
import re
import os.path


class MyWebCrawler:
    def __init__(self):
        self.to_visit = []

    def extractWords(self):
        ingredients = []
        r = urllib.request.urlopen(self.url).read()
        soup = BeautifulSoup(r)
        htmlElements = soup.find_all("dd", class_="ingredient")

        for i in range(0, len(htmlElements) - 1):
            ingredients.append(htmlElements[i].a.get_text())

        return ingredients

    def exploreTree(self, url, to_vis, e, fp):
        try:
            html_page = urllib.request.urlopen(url)
            soup = BeautifulSoup(html_page)
            links = soup.findAll('a', attrs={'href': re.compile("^http://.*giallozafferano\.it")})

            for link in links:
                if link.get("href") not in self.to_visit:
                    self.to_visit.append(link.get("href"))
                    fp.write(link.get("href") + "\n")
                    print(link.get("href"))
            self.exploreTree(self.to_visit[e], self.to_visit[e], e + 1, fp)
        except Exception:
            print("Sono stato bloccato, sto in attesa 10 secondi....")
            sleep(10)
            self.exploreTree(self.to_visit[e], self.to_visit[e], e + 1, fp)
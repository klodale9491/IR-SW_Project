from bs4 import BeautifulSoup
from time import sleep
from urllib.request import HTTPError
import urllib.request
import re


class MyWebCrawler:
    def __init__(self):
        self.to_visit = []

    def getIngredients(self):
        ingredients = []
        r = urllib.request.urlopen(self.url).read()
        soup = BeautifulSoup(r)
        htmlElements = soup.find_all("dd", class_="ingredient")

        for el in  htmlElements:
            ingredients.append(el.a.get_text())

        return ingredients

    def getSitemap(self, url, e, fp):
        try:
            html_page = urllib.request.urlopen(url)
            soup = BeautifulSoup(html_page)
            links = soup.findAll('a', attrs={'href': re.compile("^http://.*giallozafferano\.it")})

            for link in links:
                if link.get("href") not in self.to_visit:
                    self.to_visit.append(link.get("href"))
                    fp.write(link.get("href") + "\n")
                    print(link.get("href"))
            self.exploreTree(self.to_visit[e], e + 1, fp)
        except Exception:
            print("Waiting....")
            sleep(10)
            self.exploreTree(self.to_visit[e], e + 1, fp)


    def getSitemap2(self,url):
        fp = open("links2.txt","w")
        coda = []
        all_links = []
        fp.write(url+"\n")
        coda.append(url)

        while len(coda) >= 1:
            try:
                html_page = urllib.request.urlopen(coda[0])
                soup = BeautifulSoup(html_page)
                links = soup.findAll('a', attrs={'href': re.compile("^http://.*giallozafferano\.it")})

                for link in links:
                    text_link = link.get("href")
                    if text_link not in all_links:
                        all_links.append(text_link)
                        coda.append(text_link)
                        fp.write(text_link + "\n")
                        print(text_link)
                # Rimuovo dalla coda il link radice
                del(coda[0])

            except HTTPError as Error:
                if Error.code == 404:
                    del(coda[0])
                    continue
                if Error.code == 502:
                    print("Waiting...")
                    sleep(10)



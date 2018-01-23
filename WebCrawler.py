from bs4 import BeautifulSoup
from time import sleep
from urllib.request import HTTPError
import urllib.request
import re


class MyWebCrawler:
    def __init__(self):
        pass

    def setRicetteLinks(self,url):
        coda = []
        all_links = []
        max_attempts = 10
        count_attempt = 1

        fp = open("links_ricetta.txt", "w")
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
                        if re.match("^http://ricette\.giallozafferano\.it",text_link):
                            fp.write(text_link + "\n")
                        print(text_link)
                # Rimuovo dalla coda il link radice
                del(coda[0])
            except HTTPError as Error:
                if Error.code != 502:
                    del(coda[0])
                    continue
                else :
                    if count_attempt < max_attempts:
                        print("Waiting " + str(count_attempt * 10) + "seconds...\n")
                        sleep(count_attempt * 10)
                        count_attempt += 1
                    else:
                        break
            except Exception:
                del(coda[0])
                continue




import re
import urllib.request
from time import sleep
from urllib.request import HTTPError

import mysql.connector
from bs4 import BeautifulSoup

from Ricetta import Ricetta


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
                else:
                    if count_attempt < max_attempts:
                        print("Waiting " + str(count_attempt * 10) + "seconds...\n")
                        sleep(count_attempt * 10)
                        count_attempt += 1
                    else:
                        break
            except Exception:
                del(coda[0])
                continue

    def loadDBRicette(self):
        mysql_config = {
            'user': 'root',
            'password': '',
            'host': '127.0.0.1',
            'database': 'giallo_zafferano',
            'raise_on_warnings': True,
        }
        cnx = mysql.connector.connect(**mysql_config, buffered=True)
        crs = cnx.cursor()
        with open("link_ricette.txt","r") as fp:
            for url in fp:
                print("Loading " + url + "\n")
                ric = Ricetta(url)
                # Aggiungo dati della ricetta
                add_ricetta = "INSERT INTO ricette(link,category,subcategory) VALUES(%s,%s,%s)"
                dati_ricetta = (url,ric.category,ric.subCategory)
                crs.execute(add_ricetta,dati_ricetta)
                ric_id = crs.lastrowid
                cnx.commit()
                # Aggiungo gli ingredienti della ricetta
                if ric.ingredients != None:
                    for ing in ric.ingredients:
                        try:
                            crs.execute("select id from ingredienti where nome = %s and link = %s",ing)
                            if crs._rowcount > 0:
                                ing_id = crs.fetchone()[0]
                            else:
                                add_ingrediente = "INSERT INTO ingredienti(nome,link) VALUES(%s,%s)"
                                dati_ingrediente = ing
                                crs.execute(add_ingrediente,dati_ingrediente)
                                cnx.commit()
                                ing_id = crs.lastrowid
                            add_ingrediente_ricetta = "INSERT INTO ingredienti_ricette(id_ricetta,id_ingrediente) VALUES(%s,%s)"
                            dati_ingrediente_ricetta = (ric_id,ing_id)
                            crs.execute(add_ingrediente_ricetta, dati_ingrediente_ricetta)
                            cnx.commit()
                        except mysql.connector.Error as Err:
                            if Err.errno == 1062:
                                continue

        cnx.close()
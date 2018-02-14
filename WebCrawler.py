import re
import urllib.request
import mysql.connector
from time import sleep
from urllib.request import HTTPError
from bs4 import BeautifulSoup
from Ricetta import Ricetta
from DBConnector import DBConnector


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
        cnx = DBConnector.connect('root', 'root', '127.0.0.1', 'giallo_zafferano')
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
                            if Err.errno == 1062: # Violazione della chiave composta UNIQUE nome_link
                                continue
                            else:
                                print(Err.msg)
                                continue
                # Aggiungo gli step di preparazione della ricetta
                if ric.prep != None:
                    len_step_prep = len(ric.prep)
                    if len_step_prep > 0:
                        tup_lst = list()
                        query = "INSERT INTO preparazioni_ricette(id_ricetta,step,descrizione_step) VALUES(%s,%s,%s)"
                        step = 0
                        for index in range(0,len_step_prep-1):
                            tup_lst.append((ric_id,step,ric.prep[index],))
                            step += 1
                        tup_lst.append((ric_id, step, ric.prep[step],))
                        try:
                            crs.executemany(query,tup_lst)
                        except mysql.connector.Error as Err:
                            print(Err.msg)
                            continue
        cnx.close()

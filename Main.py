from WebCrawler import MyWebCrawler
from Ricetta import Ricetta

r = Ricetta("http://ricette.giallozafferano.it/Limoni-ripieni-di-crema-al-tonno.html")
print(r.prep)
print(r.ingredients)
print(r.subCategory)
print(r.category)
#we = MyWebCrawler()
#fp = open("links.txt","w")
# we.getSitemap("http://ricette.giallozafferano.it/",0,fp)
#we.getSitemap2("http://ricette.giallozafferano.it/")
#print(we.to_visit)
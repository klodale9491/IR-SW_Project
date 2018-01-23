from WebCrawler import MyWebCrawler

'''
r = Ricetta("http://ricette.giallozafferano.it/Fagottini-di-pasta-sfoglia-con-cuore-di-mela.html")
print(r.prep)
print(r.ingredients)
print(r.category)
print(r.subCategory)
'''

wc = MyWebCrawler()
wc.loadDBRicette()
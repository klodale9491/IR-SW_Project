from WebCrawler import MyWebCrawler

'''
r = Ricetta("http://ricette.giallozafferano.it/Fagottini-di-pasta-sfoglia-con-cuore-di-mela.html")
print(r.prep)
print(r.ingredients)
print(r.category)
print(r.subCategory)
'''


'''
import treetaggerwrapper
from pprint import pprint


it_string = "Tonno sott'olio sgocciolato"
it_string = it_string.replace("\t","")
tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
tags = tagger.tag_text(it_string)
pprint(treetaggerwrapper.make_tags(tags))
'''

wc = MyWebCrawler()
wc.loadDBRicette()
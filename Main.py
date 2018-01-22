from WebCrawler import MyWebCrawler


we = MyWebCrawler()
fp = open("links.txt","w")
we.exploreTree("http://ricette.giallozafferano.it/",[],0,fp)
print(we.to_visit)
#ingredienti = we.extractWords()
#print(ingredienti)
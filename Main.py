from WebCrawler import MyWebCrawler


we = MyWebCrawler()
fp = open("links.txt","w")
# we.getSitemap("http://ricette.giallozafferano.it/",0,fp)
we.getSitemap2("http://ricette.giallozafferano.it/")
print(we.to_visit)
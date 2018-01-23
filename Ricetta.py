import urllib.request
from bs4 import BeautifulSoup



class Ricetta:
    def __init__(self, url):
        self.content = BeautifulSoup(urllib.request.urlopen(url).read())
        self.setCategory()
        self.setIngredients()
        self.setPreparation()
        self.setSubCategory()

    def setIngredients(self):
        self.ingredients = self.content.find_all("dd", class_="ingredient")

    def setCategory(self):
        self.category = self.content.find_all("a", class_="rkat")

    def setSubCategory(self):
        subCategory = self.content.find_all("a", class_="main-category")
        if len(subCategory):
            self.subCategory = subCategory
        else:
            self.subCategory = None

    def setPreparation(self):
        node = self.content.find_all("div", class_="sez mtop")[0]
        prep = []
        while node is not None:
            #if type(node) == ''
            if node.name == 'h2' and 'sez' in node.attrs['class']:
                break
            node = node.next_sibling
            if node.name == 'p':
                prep.append(node.text)
        self.prep = prep

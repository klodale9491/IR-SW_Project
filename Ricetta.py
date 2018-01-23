import urllib.request
from urllib.request import URLError

from bs4 import BeautifulSoup


class Ricetta:
    def __init__(self, url):
        try:
            self.content = BeautifulSoup(urllib.request.urlopen(url).read())
            self.setCategory()
            self.setSubCategory()
            self.setIngredients()
            self.setPreparation()
        except URLError:
            self.category = None
            self.subCategory = None
            self.ingredients = None
            self.prep = None

    def setIngredients(self):
        self.ingredients = list()
        ingredient_tags = self.content.find_all("dd", class_="ingredient")
        for tag in ingredient_tags:
            ingredient = tag.text
            # Pulizia delle stringhe degli ingredienti
            #for ch in ['\n']:
            #   ingredient = ingredient.replace('','\t')
            self.ingredients.append(ingredient)

    def setCategory(self):
        category_tag = self.content.find("a", class_="rkat")
        if category_tag != None:
            self.category = category_tag.text
        else:
            self.category = None

    def setSubCategory(self):
        subCategory_tag = self.content.find("a", class_="main-category")
        if subCategory_tag != None:
            self.subCategory = subCategory_tag.text
        else:
            self.subCategory = None

    def setPreparation(self):
        node = self.content.find("div", class_="sez mtop")
        prep = []
        while node is not None:
            #if type(node) == ''
            if node.name == 'h2' and 'sez' in node.attrs['class']:
                break
            if node.name == 'p' and node.text != '':
                phrases = node.text.split('.')
                for phrase in phrases:
                    prep.append(phrase)
            node = node.next_sibling
        self.prep = prep
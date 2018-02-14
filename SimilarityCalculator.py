import math
from numpy import inf

from DBConnector import DBConnector


class SimilarityCalculator:
    def __init__(self):
        self.setMatrix(self.calculateMatrix())
        self.pmi()

    def pmi(self):
        self.pmi_px()
        self.pmi_pxy()
        num_ingr = len(self.matrix[0])
        self.pmi_matrix = [[0 for x in range(num_ingr)] for y in range(num_ingr)]
        for i in range(0,num_ingr):
            for j in range(0,num_ingr):
                t = math.log(self.pxy[i][j]) if self.pxy[i][j] != 0 else -inf
                self.pmi_matrix[i][j] = t - math.log(self.px[i]) - math.log(self.px[j])

    def pmi_px(self):
        num_ingr = len(self.matrix[0])
        self.px = [0 for x in range(num_ingr)]
        num_ricette = len(self.matrix)
        for i in range(0, num_ingr):
            for j in range(0, num_ricette):
                if self.matrix[j][i] == 1:
                    self.px[i] = self.px[i] + 1
        for i in range(0, num_ingr):
            self.px[i] = self.px[i] / num_ricette

    def pmi_pxy(self):
        num_ingr = len(self.matrix[0])
        num_ric = len(self.matrix)
        self.pxy = [[0 for x in range(num_ingr)] for y in range(num_ingr)]
        for i in range(0,len(self.pxy)):
            for j in range(0,i+1):
                c = 0
                for r in range(0,num_ric):
                    if self.matrix[r][j] == 1 and self.matrix[r][i] == 1:
                        c = c + 1
                self.pxy[i][j] = c/len(self.matrix)
                self.pxy[j][i] = self.pxy[i][j]

    def calculateMatrix(self):
        cnx = DBConnector().connect('root', 'gzhzvzx', '127.0.0.1', 'giallo_zafferano')
        crs = cnx.cursor()
        crs.execute("select count(*) as c from ingredienti")
        count_ingr = crs.fetchone()[0]
        count_ingr = 49
        crs.execute("select count(*) as c from ricette")
        #crs.execute("select count(distinct id_ricetta) from ingredienti_ricette where id_ingrediente < 50")
        count_ric = crs.fetchone()[0]
        crs.execute("select * from ingredienti_ricette where id_ingrediente < 50")
        matrix = [[0 for x in range(count_ingr)] for y in range(count_ric)]
        row = crs.fetchone()
        l = list()
        while row is not None:
            if row[1] not in l:
                l.append(row[1])
                r = len(l)-1
            matrix[r][row[2]-1] = 1
            row = crs.fetchone()
        return matrix

    def getMatrix(self):
        return self.matrix

    def setMatrix(self, matrix):
        self.matrix = matrix

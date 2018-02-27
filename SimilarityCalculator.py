import math

from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine as cosine_distance

from DBConnector import DBConnector


class SimilarityCalculator:
    def __init__(self):
        self.setMatrix(self.calculateMatrix())
        self.pmi_px()
        self.pmi_pxy()
        self.pmi()
        self.sort_pmi()
        self.pmi()
        self.b_pmi()
        # self.cosine_distance()

    def pmi(self):
        print("pmi")
        self.pmi_matrix = [[0 for x in range(self.num_ingr)] for y in range(self.num_ingr)]
        for i in range(0,self.num_ingr):
            for j in range(0,self.num_ingr):
                if self.pxy[i][j] > 0:
                    self.pmi_matrix[i][j] = math.log(self.pxy[i][j] * self.m, 2) - math.log(self.px[i], 2) - math.log(self.px[j], 2)
        print("DONE")

    def b_pmi(self):
        print("b_pmi")
        self.bpmi = [[0 for x in range(self.num_ingr)] for y in range(self.num_ingr)]
        sigma = 6.5
        eps = 3
        for i in range(0, len(self.bpmi)):
            bi = round(math.pow(math.log(self.px[i]),2) * math.log(self.num_ingr,2) / sigma) + 1
            for j in range(0, i + 1):
                bj = round(math.pow(math.log(self.px[j]), 2) * math.log(self.num_ingr, 2) / sigma) + 1
                sum_i = 0
                for x in range(0, bi):
                    if x >= self.num_ingr - 1:
                        break
                    sum_i = sum_i + math.pow(self.pmi_matrix[self.sort[i][x]][j], eps)
                sum_j = 0
                for y in range(0, bj):
                    if y >= self.num_ingr - 1:
                        break
                    sum_j = sum_j + math.pow(self.pmi_matrix[self.sort[j][x]][i], eps)
                self.bpmi[i][j] = sum_i / bi + sum_j / bj
                self.bpmi[j][i] = self.bpmi[i][j]
                #if bi == 1 or bj == 1:
                #    print(i, j, bi, bj, self.px[i], self.px[j], self.bpmi[i][j])
        print("DONE")

    def sort_pmi(self):
        print("sort_pmi")
        self.sort = self.pmi_matrix
        for i in range(0, len(self.pmi_matrix)):
            self.sort[i] = sorted(range(len(self.pmi_matrix[i])), key=self.pmi_matrix[i].__getitem__, reverse=True)
        print("DONE")

    def pmi_px(self):
        print("pmi_px")
        self.px = [0 for x in range(self.num_ingr)]
        num_ricette = len(self.matrix)
        for i in range(0, self.num_ingr):
            for j in range(0, num_ricette):
                if self.matrix[j][i] == 1:
                    self.px[i] = self.px[i] + 1
        print("DONE")
        #for i in range(0, num_ingr):
        #    self.px[i] = self.px[i] / num_ricette

    def pmi_pxy(self):
        print("pmi_pxy")
        self.pxy = [[0 for x in range(self.num_ingr)] for y in range(self.num_ingr)]
        cnx = DBConnector().connect('root', 'root', '127.0.0.1', 'giallo_zafferano')
        crs = cnx.cursor()
        crs.execute("select distinct a.id_ricetta, a.id_ingrediente, b.id_ingrediente from ingredienti_ricette a, ingredienti_ricette b where a.id_ricetta = b.id_ricetta and a.id_ingrediente > b.id_ingrediente and a.id_ingrediente < 50")
        row = crs.fetchone()
        while row is not None:
            self.pxy[row[1] - 1][row[2] - 1] = self.pxy[row[1] - 1][row[2] - 1] + 1
            self.pxy[row[2] - 1][row[1] - 1] = self.pxy[row[2] - 1][row[1] - 1] + 1
            row = crs.fetchone()
        '''g = 0
        for i in range(0,len(self.pxy)):
            for j in range(0,i+1):
                c = 0
                if i!=j:
                    for r in range(0,self.num_ric):
                        if self.matrix[r][j] == 1 and self.matrix[r][i] == 1:
                            c = c + 1
                            g = g + 1
                self.pxy[i][j] = c
                self.pxy[j][i] = c
        print(g)'''
        print("DONE")

    def calculateMatrix(self):
        cnx = DBConnector().connect('root', 'root', '127.0.0.1', 'giallo_zafferano')
        crs = cnx.cursor()
        crs.execute("select count(*) as c from ingredienti")
        self.num_ingr = crs.fetchone()[0]
        #count_ingr = 49
        crs.execute("select count(*) as c from ricette")
        #crs.execute("select count(distinct id_ricetta) from ingredienti_ricette where id_ingrediente < 50")
        self.num_ric = crs.fetchone()[0]
        #crs.execute("select * from ingredienti_ricette where id_ingrediente < 50")
        crs.execute("select * from ingredienti_ricette")
        self.m = crs.rowcount
        matrix = [[0 for x in range(self.num_ingr)] for y in range(self.num_ric)]
        row = crs.fetchone()
        self.l = list()
        while row is not None:
            if row[1] not in self.l:
                self.l.append(row[1])
                r = len(self.l)-1
            matrix[r][row[2]-1] = 1
            row = crs.fetchone()
        return matrix

    # Calculate TSVD Matrix Decomposition
    def cosine_distance(self,comp):
        print('cosine_distance')
        self.svd = TruncatedSVD(n_components=comp, n_iter=7, random_state=42).fit_transform(self.matrix)
        self.cosine_matrix = [[0 for x in range(self.num_ric)] for y in range(self.num_ric)]
        #self.svd = self.matrix
        for i in range(0,len(self.svd)):
            for j in range(i+1,len(self.svd)):
                cosine = cosine_distance(self.svd[i],self.svd[j])
                self.cosine_matrix[i][j] = cosine
                self.cosine_matrix[j][i] = cosine
        print("DONE")


    def getMatrix(self):
        return self.matrix

    def setMatrix(self, matrix):
        self.matrix = matrix

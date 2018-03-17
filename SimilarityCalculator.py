import math
import json
import numpy
import multiprocessing
import time

from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine as cosine_distance
from numpy import argsort
from DBConnector import DBConnector


class SimilarityCalculator:
    def __init__(self,num_proc):
        self.num_proc = num_proc
        self.init_var()
        self.calculate_matrix_sql()
        self.mem_alloc()
        self.par_ops()
        self.show_results(self.bsort,self.bpmi)
        #self.save_results_sql(self.bpmi)
        '''
        self.pmi_px()
        self.pmi_pxy_2()
        self.pmi()
        self.sort_pmi()
        self.pmi()
        self.b_pmi()
        self.sort_bpmi()
        self.b_pmi()
        '''

    # Init variables
    def init_var(self):
        cnx = DBConnector().connect()
        crs = cnx.cursor()
        crs.execute("select count(*) as c from ingredienti")
        self.num_ingr = crs.fetchone()[0]
        crs.execute("select count(*) as c from ricette")
        self.num_ric = crs.fetchone()[0]

    # Allocate shared memory for operations
    def mem_alloc(self):
        print("mem_alloc")
        self.px = multiprocessing.Array("i",self.num_ingr)
        self.pxy = multiprocessing.Array("i",self.num_ingr*self.num_ingr)
        self.pmi_matrix = multiprocessing.Array("d",self.num_ingr*self.num_ingr)
        self.sort = multiprocessing.Array("i",self.num_ingr*self.num_ingr)
        self.bpmi = multiprocessing.Array("d",self.num_ingr*self.num_ingr)
        self.bsort = multiprocessing.Array("i",self.num_ingr*self.num_ingr)
        # Lut bpmi
        self.b = multiprocessing.Array("i",self.num_ingr)
        print("DONE")

    # Parallelize operations
    def par_ops(self):
        my_procs = [None for x in range(self.num_proc)]
        # Parallelize pmi_px
        print("pmi_px")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.pmi_px, args=(i,self.px))
            my_procs[i].start()
            time.sleep(0.1)
        for j in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize pmi_pxy_2
        print("pmi_pxy_2")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.pmi_pxy_2, args=(i,self.pxy))
            my_procs[i].start()
            time.sleep(0.1)
        for j in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize pmi
        print("pmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.pmi, args=(i,self.px,self.pxy,self.pmi_matrix))
            my_procs[i].start()
            time.sleep(0.1)
        for j in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize sort_pmi
        print("sort_pmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.sort_pmi, args=(i,self.sort,self.pmi_matrix))
            my_procs[i].start()
            time.sleep(0.1)
        for j in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize pmi
        print("pmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.pmi, args=(i,self.px,self.pxy,self.pmi_matrix))
            my_procs[i].start()
            time.sleep(0.1)
        for j in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize b_pmi
        print("bpmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.b_pmi, args=(i,self.b,self.px,self.pmi_matrix,self.sort,self.bpmi))
            my_procs[i].start()
            time.sleep(0.1)
        for j in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize sort_bpmi
        print("sort_bpmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.sort_bpmi, args=(i,self.bsort,self.bpmi))
            my_procs[i].start()
            time.sleep(0.1)
        for j in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize b_pmi
        print("bpmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.b_pmi,args=(i, self.b, self.px, self.pmi_matrix, self.sort, self.bpmi))
            my_procs[i].start()
            time.sleep(0.1)
        for j in range(self.num_proc):
            my_procs[i].join()
        print("DONE")

    # Calculate P(x),P(y)
    def pmi_px(self,tid,my_px):
        px = numpy.frombuffer(my_px.get_obj(),numpy.int32)
        num_ricette = len(self.matrix)
        start = tid
        for i in range(start, self.num_ingr, self.num_proc):
            for j in range(0, num_ricette):
                if self.matrix[j][i] == 1:
                    px[i] += 1

    # Calculate P(x,y) using database
    def pmi_pxy(self):
        self.pxy = [[0 for x in range(self.num_ingr)] for y in range(self.num_ingr)]
        cnx = DBConnector().connect()
        crs = cnx.cursor()
        crs.execute("select distinct a.id_ricetta, a.id_ingrediente, b.id_ingrediente from ingredienti_ricette a, ingredienti_ricette b where a.id_ricetta = b.id_ricetta and a.id_ingrediente > b.id_ingrediente")
        row = crs.fetchone()
        while row is not None:
            self.pxy[row[1] - 1][row[2] - 1] = self.pxy[row[1] - 1][row[2] - 1] + 1
            self.pxy[row[2] - 1][row[1] - 1] = self.pxy[row[2] - 1][row[1] - 1] + 1
            row = crs.fetchone()

    # Calculate P(x,y) in memory
    def pmi_pxy_2(self,tid,pxy):
        start = tid
        pxy = numpy.frombuffer(pxy.get_obj(), numpy.int32).reshape(self.num_ingr,self.num_ingr)

        for i in range(start,self.num_ric,self.num_proc):
            for j in range(self.num_ingr):
                if self.matrix[i][j] == 1:
                     for k in range(j):
                         if self.matrix[i][k] == 1:
                             pxy[j][k] += 1
                             pxy[k][j] += 1

    # Calculate PMI(x,y) = log (P(x,y)/P(x)*P(y))
    def pmi(self,tid,px,pxy,pmi_matrix):
        start = tid
        px = numpy.frombuffer(px.get_obj(), numpy.int32)
        pxy = numpy.frombuffer(pxy.get_obj(), numpy.int32).reshape(self.num_ingr,self.num_ingr)
        pmi_matrix = numpy.frombuffer(pmi_matrix.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_ingr)

        for i in range(start,self.num_ingr,self.num_proc):
            for j in range(0,i + 1):
                if pxy[i][j] > 0:
                    pmi_matrix[i][j] = math.log(pxy[i][j] * self.num_ric / (px[i] * px[j]), 2)
                    pmi_matrix[j][i] = pmi_matrix[i][j]
                    #self.npmi[i][j] = math.log(self.px[i] * self.px[j] / pow(self.num_ric,2)) / math.log(self.pxy[i][j] / self.num_ric) - 1
                #else:
                    # self.npmi[i][j] = -1
                    # self.pmi_matrix[i][j] = -10

    # Calculate second order PMI
    def b_pmi(self,tid,b,px,pmi_matrix,sort,bpmi):
        sigma = 6.5
        eps = 3

        start = tid
        b = numpy.frombuffer(b.get_obj(), numpy.int32)
        px = numpy.frombuffer(px.get_obj(), numpy.int32)
        pmi_matrix = numpy.frombuffer(pmi_matrix.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_ingr)
        sort = numpy.frombuffer(sort.get_obj(), numpy.int32).reshape(self.num_ingr,self.num_ingr)
        bpmi = numpy.frombuffer(bpmi.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_ingr)

        # LUT to speedup computing
        for i in range(0, self.num_ingr):
            b[i] = round(math.pow(math.log(px[i]), 2) * math.log(self.num_ingr, 2) / sigma) + 1
        # Parallelize cycle...
        for i in range(start, len(bpmi), self.num_proc):
            # bi = round(math.pow(math.log(self.px[i]), 2) * math.log(self.num_ingr, 2) / sigma) + 1
            for j in range(0, i + 1):
                if j % 100  == 0 and i % 100 == 0:
                    print(i,j)
                # bj = round(math.pow(math.log(self.px[j]), 2) * math.log(self.num_ingr, 2) / sigma) + 1
                sum_i = 0
                for x in range(0, b[i]):
                    if x >= self.num_ingr - 1:
                        break
                    if pmi_matrix[sort[i][x]][i] > 0 and pmi_matrix[sort[i][x]][j] > 0 and sort[i][x] in sort[j][0:b[j]]:
                        sum_i = sum_i + math.pow(pmi_matrix[sort[i][x]][j], eps)
                sum_j = 0
                for y in range(0, b[j]):
                    if y >= self.num_ingr - 1:
                        break
                    if pmi_matrix[sort[j][y]][i] > 0 and pmi_matrix[sort[j][y]][j] > 0 and sort[j][y] in sort[i][0:b[i]]:
                        sum_j = sum_j + math.pow(pmi_matrix[sort[j][y]][i], eps)
                if i in sort[j][0:b[j]] or j in sort[i][0:b[i]]:
                    bpmi[i][j] = 0
                    bpmi[j][i] = 0
                else:
                    bpmi[i][j] = sum_i / b[i] + sum_j / b[j]
                    bpmi[j][i] = bpmi[i][j]

    def sort_pmi(self,tid,sort,pmi_matrix):
        start = tid
        sort = numpy.frombuffer(sort.get_obj(), numpy.int32).reshape(self.num_ingr,self.num_ingr)
        pmi_matrix = numpy.frombuffer(pmi_matrix.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_ingr)
        for i in range(start, len(pmi_matrix),self.num_proc):
            sort[i] = numpy.argsort(pmi_matrix[i])[::-1]

    def sort_bpmi(self,tid,bsort,bpmi):
        start = tid
        bsort = numpy.frombuffer(bsort.get_obj(), numpy.int32).reshape(self.num_ingr,self.num_ingr)
        bpmi = numpy.frombuffer(bpmi.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_ingr)
        for i in range(start, len(bpmi), self.num_proc):
            bsort[i] = numpy.argsort(bpmi[i])[::-1]

    def sort_npmi(self,tid,sort):
        start = tid
        self.nsort = self.npmi
        for i in range(start, len(self.npmi),self.num_proc):
            self.nsort[i] = argsort(self.npmi[i])[::-1]

    # Calculate cosine distance truncating occurrence matrix using TSVD decomposition
    def cosine_distance(self, comp=None):
        print('cosine_distance')
        if comp is None:
            self.svd = self.matrix
        else:
            self.svd = TruncatedSVD(n_components=comp, n_iter=7, random_state=42).fit_transform(self.matrix)
        self.cosine_matrix = [[0 for x in range(self.num_ric)] for y in range(self.num_ric)]
        #self.svd = self.matrix
        for i in range(0,len(self.svd)):
            for j in range(i+1,len(self.svd)):
                cosine = cosine_distance(self.svd[i],self.svd[j])
                self.cosine_matrix[i][j] = cosine
                self.cosine_matrix[j][i] = cosine
        print("DONE")

    def calculate_matrix_sql(self):
        print("calculate_matrix_sql")
        cnx = DBConnector().connect()
        crs = cnx.cursor()
        #crs.execute("select ingredienti_ricette.id, ingredienti_ricette.id_ricetta, ingredienti_ricette.id_ingrediente, ingredienti.nome from ingredienti_ricette, ingredienti where ingredienti.id_ing = ingredienti_ricette.id_ingrediente order by ingredienti_ricette.id_ricetta")
        crs.execute("select ingredienti_ricette.id, ingredienti_ricette.id_ricetta, ingredienti_ricette.id_ingrediente, ingredienti.nome from ingredienti_ricette, ingredienti where ingredienti.id = ingredienti_ricette.id_ingrediente order by ingredienti_ricette.id_ricetta")
        self.m = crs.rowcount
        matrix = multiprocessing.Array("d",self.num_ingr*self.num_ric)
        self.matrix = numpy.frombuffer(matrix.get_obj()).reshape(self.num_ric,self.num_ingr)
        del(matrix)
        row = crs.fetchone()
        self.l = list()
        self.i = [0 for x in range(self.num_ingr)]
        while row is not None:
            if row[1] not in self.l:
                self.l.append(row[1])
                r = len(self.l)-1
            self.matrix[r][row[2]-1] = 1
            self.i[row[2]-1] = row[3]
            row = crs.fetchone()
        print("DONE")

    # Save results into results.dat file
    def show_results(self,bsort,bpmi):
        print("show_results")
        fout = open("results.dat", "w")
        num_elm = 10
        cnx = DBConnector().connect()
        crs = cnx.cursor()
        bsort = numpy.frombuffer(bsort.get_obj(), numpy.int32).reshape(self.num_ingr, self.num_ingr)
        bpmi = numpy.frombuffer(bpmi.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_ingr)
        for i in range(self.num_ingr):
            #Per ogni ingredediente prendo i 10 più simili
            best_sim = bsort[i][0:num_elm]
            crs.execute("select nome from ingredienti where id = %s",(i+1,))
            cnx.commit()
            nome_ing = crs.fetchone()[0]
            fout.write("Ingrediente : " + str(nome_ing)+"\n")
            # Preleviamo quelli più simili
            for j in range(num_elm):
                crs.execute("select nome from ingredienti where id = %s",(int(bsort[i][j]+1),))
                cnx.commit()
                fout.write(str(j)+")"+str(crs.fetchone()[0])+"\t"+str(bpmi[i][best_sim[j]])+"\n")
            fout.write("\n\n")
        fout.close()
        print("DONE")

    def calculate_matrix_json(self,filename):
        print("calculate_matrix_json")
        # Load data from file
        f = open(filename, "r")
        data = f.read()
        jsondata = json.loads(data)
        MAX = 20000
        # Convert dataset unicode to string
        self.recipes = []
        self.ingredients = []
        index = 0
        for urec in jsondata:
            singredients = []
            for uing in urec['ingredients']:
                # sing = conv_unicode(uing)
                singredients.append(uing)
                if uing not in self.ingredients:
                    self.ingredients.append(uing)
            self.recipes.append([index, singredients])
            index += 1
        # Get Occurrence Matrix
        self.num_ric = len(self.recipes)
        self.num_ingr = len(self.ingredients)
        self.matrix = [[0 for x in range(self.num_ingr)] for y in range(self.num_ric)]
        for ric in self.recipes:
            id_rec = ric[0]
            for ing in ric[1]:
                id_ingr = self.ingredients.index(ing)
                self.matrix[id_rec][id_ingr] = 1
        print("DONE")

    def save_results_sql(self,bpmi):
        print("save_results_sql")
        bpmi = numpy.frombuffer(bpmi.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_ingr)
        count = 0;
        cnx = DBConnector().connect()
        crs = cnx.cursor()
        for i in range(0, len(self.bpmi)):
            for j in range(0, i+1):
                crs.execute("insert into bpmi values(%s,%s,%s)",(i+1,j+1,float(bpmi[i][j])))
                count = count + 1
                if count == 1000:
                    cnx.commit()
                    count = 0
        if count != 0:
            cnx.commit()
        print("DONE")

    def get_bpmi_from_sql(self):
        cnx = DBConnector().connect()
        crs = cnx.cursor()
        crs.execute("select count(*) as c from bpmi")
        count = crs.fetchone()[0]
        for i in range(0,math.ceil(count/1000000)):
            crs.execute("select * from bpmi limit 1000000 offset %s", (i * 1000000,))

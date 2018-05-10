import math
import numpy
import multiprocessing
import time

from scipy.spatial.distance import cosine as cosine_distance
from sklearn.utils.extmath import randomized_svd
from numpy import argsort
from DBConnector import DBConnector


class SimilarityCalculator:
    def __init__(self,num_proc):
        self.num_proc = num_proc
        self.init_var()
        self.mem_alloc()
        self.occ_matrix_sql(self.matrix)
        self.par_ops()
        # Save results of similarity using : occ_matrix
        occ_sim = self.mat_sim(self.matrix, numpy.int32,self.num_ric,self.num_ingr)
        self.sav_sim_ing('results/occ_sim.dat', occ_sim, 10)
        # Save results of similarity using : pmi,bpmi,pxy
        pxy_sim = self.mat_sim(self.pxy, numpy.int32,self.num_ingr,self.num_ingr)
        self.sav_sim_ing('results/pxy_sim.dat', pxy_sim, 10)
        pmi_sim = self.mat_sim(self.pmi_matrix, numpy.float,self.num_ingr,self.num_ingr)
        self.sav_sim_ing('results/pmi_sim.dat', pmi_sim, 10)
        bpmi_sim = self.mat_sim(self.bpmi, numpy.float,self.num_ingr,self.num_ingr)
        self.sav_sim_ing('results/bpmi_sim.dat', bpmi_sim, 10)
        nacuc_sim = self.svd_sqrt_cos(self.nacuc)
        self.sav_sim_ing('results/nacuc_sim.dat', nacuc_sim, 10)
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


    def init_var(self):
        """
        Initializes main shared variables used by the program.
        :return:
        """
        cnx = DBConnector().connect()
        crs = cnx.cursor()
        crs.execute("select count(*) as c from ingredienti")
        self.num_ingr = crs.fetchone()[0]
        crs.execute("select count(*) as c from ricette")
        self.num_ric = crs.fetchone()[0]
        crs.execute("select count(distinct category) from ricette")
        self.num_cat = crs.fetchone()[0]


    def mem_alloc(self):
        """
        Allocate shared memory for operations
        :return:
        """
        print("mem_alloc")
        self.matrix = multiprocessing.Array("i", self.num_ingr * self.num_ric)
        self.px = multiprocessing.Array("i",self.num_ingr)
        self.pxy = multiprocessing.Array("i",self.num_ingr*self.num_ingr)
        self.pmi_matrix = multiprocessing.Array("d",self.num_ingr*self.num_ingr)
        self.sort = multiprocessing.Array("i",self.num_ingr*self.num_ingr)
        self.bpmi = multiprocessing.Array("d",self.num_ingr*self.num_ingr)
        self.bsort = multiprocessing.Array("i",self.num_ingr*self.num_ingr)
        self.cat = multiprocessing.Array("d", self.num_ingr * self.num_cat)
        self.nacuc = multiprocessing.Array("d", self.num_ingr * self.num_ingr)
        # Lut bpmi
        self.b = multiprocessing.Array("i",self.num_ingr)
        print("DONE")


    def par_ops(self):
        """
        Parallelize operations.
        :return:
        """
        my_procs = [None for x in range(self.num_proc)]
        # Parallelize pmi_px
        print("pmi_px")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.pmi_px, args=(i,self.matrix,self.px))
            my_procs[i].start()
            time.sleep(0.1)
        for i in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Categories
        print("cat")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.pcat, args=(i, self.cat))
            my_procs[i].start()
            time.sleep(0.1)
        for i in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize pmi_pxy_2
        print("pmi_pxy")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.pmi_pxy, args=(i,self.matrix,self.pxy))
            my_procs[i].start()
            time.sleep(0.1)
        for i in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize pmi
        print("pmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.pmi, args=(i,self.px,self.pxy,self.pmi_matrix))
            my_procs[i].start()
            time.sleep(0.1)
        for i in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize sort_pmi
        print("sort_pmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.sort_pmi, args=(i,self.sort,self.pmi_matrix))
            my_procs[i].start()
            time.sleep(0.1)
        for i in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize pmi
        print("pmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.pmi, args=(i,self.px,self.pxy,self.pmi_matrix))
            my_procs[i].start()
            time.sleep(0.1)
        for i in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize b_pmi
        print("bpmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.b_pmi, args=(i,self.b,self.px,self.pmi_matrix,self.sort,self.bpmi, self.cat))
            my_procs[i].start()
            time.sleep(0.1)
        for i in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize sort_bpmi
        print("sort_bpmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.sort_bpmi, args=(i,self.bsort,self.bpmi))
            my_procs[i].start()
            time.sleep(0.1)
        for i in range(self.num_proc):
            my_procs[i].join()
        print("DONE")
        # Parallelize b_pmi
        print("bpmi")
        for i in range(self.num_proc):
            my_procs[i] = multiprocessing.Process(target=self.b_pmi,args=(i, self.b, self.px, self.pmi_matrix, self.sort, self.bpmi, self.cat))
            my_procs[i].start()
            time.sleep(0.1)
        for i in range(self.num_proc):
             my_procs[i].join()
        print("DONE")


    def pmi_px(self,tid,matrix,my_px):
        """
        Calculate P(x),P(y)
        :param tid: My ProcessID
        :param matrix: Reference to shared maemory occurence matrix (ingredients X recipes)
        :param my_px: Reference to shared memory occurrence array (occurence in all recipes per single ingredient)
        :return:
        """
        px = numpy.frombuffer(my_px.get_obj(),numpy.int32)
        matrix = numpy.frombuffer(matrix.get_obj(),numpy.int32).reshape(self.num_ingr,self.num_ric)
        num_ricette = len(matrix)
        start = tid
        for i in range(start, self.num_ingr, self.num_proc):
            for j in range(0, num_ricette):
                if matrix[j][i] == 1:
                    px[i] += 1


    # PCat
    def pcat(self, tid, cat):
        start = tid
        self.cat = numpy.frombuffer(cat.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_cat)
        cnx = DBConnector().connect()
        crs = cnx.cursor()
        crs.execute("select distinct category from ricette where category is not null")
        rs = crs.fetchall()
        dic = {}
        for i in range(0, len(rs)):
            dic[rs[i][0]] = i

        for i in range(start, self.num_ingr, self.num_proc):
            crs.execute("select category, count(ricette.category) from ingredienti_ricette, ricette where ingredienti_ricette.id_ricetta = ricette.id and ingredienti_ricette.id_ingrediente = %s group by ricette.category", (i+1,))
            rs = crs.fetchall()
            for category in rs:
                if category[0] is None:
                    continue
                cat_index = dic[category[0]]
                self.cat[i][cat_index] = category[1] / self.px[i]


    '''
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
    '''


    def pmi_pxy(self,tid,matrix,pxy):
        """
            Calculate Calculate P(x,y) in memory
            :param tid: My ProcessID
            :param matrix: Reference to shared maemory occurence matrix (ingredients X recipes)
            :param pxy: Reference to shared memory co-occurrence matrix
            :return:
        """
        start = tid
        pxy = numpy.frombuffer(pxy.get_obj(), numpy.int32).reshape(self.num_ingr,self.num_ingr)
        matrix = numpy.frombuffer(matrix.get_obj(), numpy.int32).reshape(self.num_ric, self.num_ingr)
        for i in range(start,self.num_ric,self.num_proc):
            for j in range(self.num_ingr):
                if matrix[i][j] == 1:
                     for k in range(j):
                         if matrix[i][k] == 1:
                             pxy[j][k] += 1
                             pxy[k][j] += 1


    def pmi(self,tid,px,pxy,pmi_matrix):
        """
        Calculate PMI(x,y) = log(P(x,y)/P(x)*P(y))
        :param tid: My ProcessID
        :param px: Reference to shared memory occurrence array (occurence in all recipes per single ingredient)
        :param pxy: Reference to shared memory co-occurrence matrix
        :param pmi_matrix: Reference to shared memory PMI matrix
        :return:
        """
        start = tid
        px = numpy.frombuffer(px.get_obj(), numpy.int32)
        pxy = numpy.frombuffer(pxy.get_obj(), numpy.int32).reshape(self.num_ingr,self.num_ingr)
        pmi_matrix = numpy.frombuffer(pmi_matrix.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_ingr)

        for i in range(start,self.num_ingr,self.num_proc):
            for j in range(0,i + 1):
                if pxy[i][j] > 0:
                    pmi_matrix[i][j] = math.log(pxy[i][j] * self.num_ric / (px[i] * px[j]), 2)
                    pmi_matrix[j][i] = pmi_matrix[i][j]


    # Calculate second order PMI
    def b_pmi(self,tid,b,px,pmi_matrix,sort,bpmi,cat):
        sigma = 6.5
        eps = 3

        start = tid
        b = numpy.frombuffer(b.get_obj(), numpy.int32)
        px = numpy.frombuffer(px.get_obj(), numpy.int32)
        pmi_matrix = numpy.frombuffer(pmi_matrix.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_ingr)
        sort = numpy.frombuffer(sort.get_obj(), numpy.int32).reshape(self.num_ingr,self.num_ingr)
        bpmi = numpy.frombuffer(bpmi.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_ingr)
        # cat = numpy.frombuffer(cat.get_obj(), numpy.dtype(float)).reshape(self.num_ingr, self.num_cat)

        # LUT to speedup computing
        for i in range(0, self.num_ingr):
            b[i] = round(math.pow(math.log(px[i]), 2) * math.log(self.num_ingr, 2) / sigma) + 1
        # Parallelize cycle...
        for i in range(start, len(bpmi), self.num_proc):
            for j in range(0, i + 1):
                if j % 100  == 0 and i % 100 == 0:
                    print(i,j)
                sum_i = 0
                for x in range(0, b[i]):
                    if x >= self.num_ingr - 1:
                        break
                    if pmi_matrix[sort[i][x]][i] > 0 and pmi_matrix[sort[i][x]][j] > 0 and sort[i][x] in sort[j][0:b[j]]:
                        # sum_i += math.pow(pmi_matrix[sort[i][x]][j], eps) * (1 - cosine_distance(cat[j], cat[sort[i][x]]))
                        sum_i += math.pow(pmi_matrix[sort[i][x]][j], eps) # Not using cosine weight
                sum_j = 0
                for y in range(0, b[j]):
                    if y >= self.num_ingr - 1:
                        break
                    if pmi_matrix[sort[j][y]][i] > 0 and pmi_matrix[sort[j][y]][j] > 0 and sort[j][y] in sort[i][0:b[i]]:
                        # sum_j += math.pow(pmi_matrix[sort[j][y]][i], eps) * (1 - cosine_distance(cat[i], cat[sort[j][y]]))
                        sum_j += math.pow(pmi_matrix[sort[j][y]][i], eps) # Not using cosine weight
                if i in sort[j][0:b[j]] or j in sort[i][0:b[i]]:
                    bpmi[i][j] = 0
                    bpmi[j][i] = 0
                else:
                    bpmi[i][j] = (sum_i / b[i] + sum_j / b[j])
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

    # Incomplete....
    def unc_cff(self,unc_mat,pxy,occ):
        """
            Calculate uncertainty matrix for ingredients.
        :return:
        """
        unc_mat = numpy.frombuffer(pxy.get_obj(), numpy.float).reshape(self.num_ingr, self.num_ingr)
        pxy = numpy.frombuffer(pxy.get_obj(), numpy.int32).reshape(self.num_ingr, self.num_ingr)
        cnt = pxy / self.num_ric # contingence/co-occurence matrix normalized
        occ = numpy.frombuffer(occ.get_obj(), numpy.int32).reshape(self.num_ric, self.num_ingr).transpose() # ingredients X recipes
        nrm_occ = numpy.array([occ[i]/math.fsum(occ[i]) for i in range(self.num_ric)]) # normalize occurences matrix
        htp = numpy.zeros((1,self.num_ingr),numpy.float) # lut entropy vector
        htp_xy = numpy.zeros((self.num_ingr,self.num_ingr),numpy.float)
        for i in range(self.num_ingr):
            htp[i] = -(math.fsum([nrm_occ[i][k] * math.log2(nrm_occ[i][k]) for k in range(len(nrm_occ[i]))]))
        for x in range(0,self.num_ingr):
            for y in range(x+1,self.num_ingr):
                pass


    def mat_sim(self,sqr_mat,mtype,dim1,dim2):
        """
        Calculate similarity between ingredients mapping them in space with less dimensions
        using TSVD.

        :param sqr_mat: It can be "pxy"(cooccurrence_matrix),"pmi",bpmi
        :param mtype: DataType of squared matrix
        :return: Similarity matrix
        """
        comps = 300
        # Using transpose to use rectangular matrix
        matrix = numpy.frombuffer(sqr_mat.get_obj(), mtype).reshape(dim1, dim2)
        if dim1 != dim2:
            matrix = matrix.transpose()
        print("matrix",matrix.shape)
        u, sigma, vt = randomized_svd(matrix,n_components=comps,n_iter=5, random_state=None)
        print("u",u.shape)
        for c in range(comps):
            u[:,c] *= math.sqrt(sigma[c]) # multiply for the weight of sigma
        mat_sim = numpy.zeros((self.num_ingr,self.num_ingr),numpy.float)
        for i in range(self.num_ingr):
            for j in range(i+1,self.num_ingr):
                mat_sim[i][j] = cosine_distance(u[i,:],u[j,:])
                mat_sim[j][i] = mat_sim[i][j]
        return mat_sim

    def sort_mat_sim(self,mat_sim):
        """
        Sort matrix by lines.

        :param mat_sim: Matrix to be sorted.
        :return: Matrix with sorted indexes..
        """
        sort_pmi_sim = numpy.zeros((self.num_ingr,self.num_ingr),numpy.int64)
        for i in range(self.num_ingr):
            sort = numpy.argsort(-mat_sim[i])[::-1]
            for j in range(len(sort)):
                sort_pmi_sim[i][j] = sort[j]
        return sort_pmi_sim

    def occ_matrix_sql(self,matrix):
        """
        Generate occurences matrix, indicaticating which ingredients appear per recipe.
        :param matrix: Reference to shared memory buffer
        :return:
        """
        print("calculate_occ_sql")
        # matrix = multiprocessing.Array("d", self.num_ingr * self.num_ric)
        matrix = numpy.frombuffer(matrix.get_obj(), numpy.int32).reshape(self.num_ric, self.num_ingr)
        cnx = DBConnector().connect()
        crs = cnx.cursor()
        #crs.execute("select ingredienti_ricette.id, ingredienti_ricette.id_ricetta, ingredienti_ricette.id_ingrediente, ingredienti.nome from ingredienti_ricette, ingredienti where ingredienti.id_ing = ingredienti_ricette.id_ingrediente order by ingredienti_ricette.id_ricetta")
        crs.execute("select ingredienti_ricette.id, ingredienti_ricette.id_ricetta, ingredienti_ricette.id_ingrediente, ingredienti.nome from ingredienti_ricette, ingredienti where ingredienti.id = ingredienti_ricette.id_ingrediente order by ingredienti_ricette.id_ricetta")
        self.m = crs.rowcount
        row = crs.fetchone()
        self.l = list()
        self.i = [0 for x in range(self.num_ingr)]
        while row is not None:
            if row[1] not in self.l:
                self.l.append(row[1])
                r = len(self.l)-1
            matrix[r][row[2]-1] = 1
            self.i[row[2]-1] = row[3]
            row = crs.fetchone()
        print("DONE")


    def sav_sim_ing(self,fname,mat_sim,num_elm=10):
        """
        Save similarity of ingredients results.
        :param fname: File where store data
        :param mat_sim: Square matrix(num_ingr X num_ingr) of "similarity"
        :param num_elm: Number of elements to be shown for each ingredient
        :return:
        """
        print("saving results...")
        fout = open(fname, "w")
        cnx = DBConnector().connect()
        crs = cnx.cursor()
        # mat_sim = numpy.frombuffer(mat_sim.get_obj(), numpy.float).reshape(self.num_ingr, self.num_ingr)
        mat_sort = self.sort_mat_sim(mat_sim)
        for i in range(self.num_ingr):
            best_sim = mat_sort[i][0:num_elm]
            crs.execute("select nome from ingredienti where id = %s",(i+1,))
            cnx.commit()
            nome_ing = crs.fetchone()[0]
            fout.write("Ingrediente : " + str(nome_ing)+"\n")
            # Get most similar
            for j in range(num_elm):
                crs.execute("select nome from ingredienti where id = %s",(int(mat_sort[i][j]+1),))
                cnx.commit()
                fout.write(str(j)+")"+str(crs.fetchone()[0])+"\t"+str(mat_sim[i][best_sim[j]])+"\n")
            fout.write("\n\n")
        fout.close()
        print("DONE")

    def svd_sqrt_cos(self, nacuc):
            pxy = numpy.frombuffer(self.pxy.get_obj(), numpy.int32).reshape(self.num_ingr, self.num_ingr)
            nacuc = numpy.frombuffer(nacuc.get_obj(), numpy.float).reshape(self.num_ingr, self.num_ingr)
            s = math.sqrt(sum(sum(pow(pxy,2))))
            nacuc = pxy / s
            comps = 300
            u, sigma, vt = randomized_svd(nacuc, n_components=comps, n_iter=5, random_state=None)
            for c in range(comps):
                u[:, c] *= math.sqrt(sigma[c])  # multiply for the weight of sigma
            mat_sim = numpy.zeros((self.num_ingr, self.num_ingr), numpy.float)
            for i in range(self.num_ingr):
                for j in range(i + 1, self.num_ingr):
                    mat_sim[i][j] = cosine_distance(u[i, :], u[j, :])
                    mat_sim[j][i] = mat_sim[i][j]
            return mat_sim
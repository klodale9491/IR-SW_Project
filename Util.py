import json
import numpy
from DBConnector import DBConnector


def store_json_sql():
    """
    Store recipe_dataset.json in SQL DB.
    :return:
    """
    print("store_json_sql")
    # Load data from file
    f = open("recipe_dataset.json", "r")
    data = f.read()
    f.close()
    jsondata = json.loads(data)
    cnx = DBConnector.connect(db="recipes_irsw")
    crs = cnx.cursor()
    for rec in jsondata:
        crs.execute("INSERT INTO ricette(id_ric,cucina) VALUES(%s,%s)", (rec['id'], rec['cuisine']))
        # Aggiungo gli ingredienti
        for ing in rec['ingredients']:
            # Verifico che l'ingrediente non sia già inserito...
            crs.execute("SELECT * FROM ingredienti WHERE id = %s",(ing,))
            row = crs.fetchone()
            if row == None:
                crs.execute("INSERT INTO ingredienti(nome) VALUES(%s)", (ing,))
                cnx.commit()
                ing_id = crs.lastrowid
            else:
                ing_id = row[0]
            # Aggiungo ingredienti ricette
            crs.execute("INSERT INTO ingredienti_ricette(id_ingrediente,id_ricetta) VALUES(%s,%s)", (ing_id, rec['id']))
            cnx.commit()
    print("DONE")

    def save_results_sql(bpmi,num_ingr):
        print("save_results_sql")
        bpmi = numpy.frombuffer(bpmi.get_obj(), numpy.dtype(float)).reshape(num_ingr,num_ingr)
        count = 0;
        cnx = DBConnector().connect()
        crs = cnx.cursor()
        for i in range(0, len(bpmi)):
            for j in range(0, i+1):
                crs.execute("insert into bpmi values(%s,%s,%s)",(i+1,j+1,float(bpmi[i][j])))
                count = count + 1
                if count == 1000:
                    cnx.commit()
                    count = 0
        if count != 0:
            cnx.commit()
        print("DONE")

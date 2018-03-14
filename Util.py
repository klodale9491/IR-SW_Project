import json
from DBConnector import DBConnector


def store_json_sql(filename):
    print("store_json_sql")

    # Load data from file
    f = open(filename, "r")
    data = f.read()
    f.close()
    jsondata = json.loads(data)

    cnx = DBConnector.connect(db="recipes_irsw")
    crs = cnx.cursor()

    for rec in jsondata:
        ric_id = rec['id']
        ric_cuisine = rec['cuisine']
        crs.execute("INSERT INTO ricette(id_ric,cucina) VALUES(%s,%s)", (ric_id, ric_cuisine))
        cnx.commit()
        # Aggiungo gli ingredienti
        for ing in rec['ingredients']:
            crs.execute("INSERT INTO ingredienti(nome) VALUES(%s)", (ing,))
            cnx.commit()
            ing_id = crs.lastrowid
            # Aggiungo ingredienti ricette
            crs.execute("INSERT INTO ingredienti_ricette(id_ingrediente,id_ricetta) VALUES(%s,%s)", (ing_id, ric_id))
            cnx.commit()
    print("DONE")




'''
cnx = DBConnector.connect(db="recipes_irsw")
crs = cnx.cursor()

# Aggiorno gli ingredeinti
crs.execute("select * from ingredienti group by nome order by id")
index = 0
data = crs.fetchall()
for row in data:
    crs.execute("UPDATE ingredienti SET id_ing = %s where nome = %s", (index, row[2]))
    cnx.commit()
    index += 1

# Aggiorno ingredienti_ricette
'''


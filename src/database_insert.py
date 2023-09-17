"""This module perform embeding data from wiki40b
   dataset into database for faster query using 
   cosine similarity."""

from time import time
import qa_utils as utils

device = utils.get_device()

wiki, conn, cursor = utils.connect_to_database()

embed_mode = utils.retrieve_model(device=device)

cur = conn.cursor()
cur.execute("select count(*) from wiki40b;")
conn.commit()
ret = cur.fetchone()[0]
print(ret)

for id, data in enumerate(wiki):
    print(id+1)
    if id >= ret:
        start = time()
        embedding_data = embed_mode.encode(data['passage_text'])

        query = f"insert into wiki40b(id, data_encoded) values ({id}, '{str(embedding_data.tolist())}');"

        cur.execute(query)
        conn.commit()
        
        end = time()
        print("time: ", end - start, " - ", end)

        if id+1 == 2000000:
            break
    
print("DONE: ")
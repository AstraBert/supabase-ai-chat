import vecs
from dotenv import load_dotenv
import os

load_dotenv()
DB_CONNECTION = os.getenv("supabase_db")

# create vector store client
vx = vecs.create_client(DB_CONNECTION)

docs = vx.get_or_create_collection(name="documents", dimension=384)

import pandas as pd
df = pd.read_csv("data/ainews_5k.csv.gz")
idex = list(df["id"])
texts = list(df["text"])
data = [{'id': idex[i], 'content': texts[i], 'embedding': []} for i in range(len(texts))]

from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

for item in data:
    docs.upsert(
        records=[
            (
            item['id'],           # the vector's identifier
            encoder.encode(item['content']).tolist(),  # the vector. list or np.array
            {"Content": item['content']}    # associated  metadata
            ),
        ]
    )


# index the collection to be queried by cosine distance
docs.create_index(measure=vecs.IndexMeasure.cosine_distance)

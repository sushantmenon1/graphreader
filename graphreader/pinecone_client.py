from pinecone import Pinecone, ServerlessSpec
import time
from tqdm import tqdm
from .embeddings import embedding_function
import os
from .utils import chunks


def push_to_pinecone(graph):
    pinecone_api_key = os.getenv('pinecone_api_key')
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = 'graphreader'

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    # connect to index
    index = pc.Index(index_name)
    time.sleep(1)

    embs = embedding_function([str(node[1]['data'])
                              for node in graph.nodes(data=True)])
    nodes = [node[0] for node in graph.nodes(data=True)]

    final_data = []
    for i in range(len(graph.nodes())):
        data = {'id': str(i), 'values': embs[i], "metadata": {
            "node": nodes[i]}}
        final_data.append(data)

    for ids_vectors_chunk in chunks(final_data, batch_size=100):
        index.upsert(vectors=ids_vectors_chunk)


def get_index():
    pinecone_api_key = os.getenv('pinecone_api_key')
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = 'graphreader'
    index = pc.Index(index_name)
    return index

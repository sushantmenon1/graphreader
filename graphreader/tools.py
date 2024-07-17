from .embeddings import get_embedding_model
from typing import List
from langchain_core.tools import tool
from .pinecone_client import get_index
import pickle
import networkx as nx


@tool
def search_tool(query: str):
    """
    Searches for the top 5 most relevant nodes based on the input query.

    This function encodes the input query into a vector representation, queries an index to find the top 5 closest matches, 
    and returns the metadata of these matches.

    Args:
        query (str): The search query string.

    Returns:
        List[dict]: A list of nodes relevant to the query.
    """
    # create the query vecto
    model = get_embedding_model()
    xq = model.encode(query).tolist()
    index = get_index()
    xc = index.query(vector=xq, top_k=5, include_metadata=True)
    return [i['metadata']['node'] for i in xc['matches']]


@tool
def read_chunk(chunk_ids: List[int]) -> str:
    """
    Concatenates and returns the document text for the given chunk IDs.

    This function retrieves the text corresponding to each chunk ID from the
    document graph and concatenates them into a single string.

    Args:
        chunk_ids (List[int]): A list of chunk IDs to be read and concatenated.

    Returns:
        str: The concatenated document text for the specified chunk IDs.
    """
    chunk_ids.extend([i-1 for i in chunk_ids] + [i+1 for i in chunk_ids])
    chunk_ids = list(set(chunk_ids))
    document = ""

    with open('chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)

    for chunk_id in chunk_ids:
        document += chunks[chunk_id]
    return document


@tool
def stop_and_read_neighbor(node_name: str) -> str:
    """
    Retrieve and read data chunks from neighboring nodes in a graph.

    Given a node name, this function identifies its neighboring nodes,
    collects the chunk IDs of data associated with those neighbors, and
    reads the chunks using the `read_chunk` function.

    Args:
        node_name (str): The name of the node whose neighbors' data chunks are to be read.

    Returns:
        str: The combined content of the chunks read from the neighboring nodes.
    """
    graph = nx.read_gml("graph.gml")
    neighbor_chunks = []
    for neighbor in list(graph.neighbors(node_name))[:2]:
        neighbor_chunks.extend(graph.nodes()[neighbor]['data'])
    return neighbor_chunks


@tool
def get_node_data(node_name: str) -> List[dict]:
    """
    Retrieve the chunk IDs associated with a given node in the graph.

    Args:
        node_name (str): The name of the node whose associated chunk IDs are to be retrieved.

    Returns:
        List[dict]: A list of dict with facts and their chunk ids.
    """
    graph = nx.read_gml("graph.gml")
    return graph.nodes()[node_name]['data']

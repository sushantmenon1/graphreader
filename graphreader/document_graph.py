import networkx as nx
import re
from collections import defaultdict
from tqdm import tqdm
from .prompts import summarize_text, generate_key_elements
import os
import pickle


class DocumentGraph:
    def __init__(self, document, max_length=1000):
        self.api_key = os.getenv("gpt_api_key")
        self.document = document
        self.max_length = max_length
        self.graph = nx.Graph()
        self.key_elements = defaultdict(set)

    def chunk_document(self):
        """Split the document into chunks of maximum length while preserving paragraph structure."""
        sentences = re.split(r'(?<=[.!?]) +', self.document)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            if len(' '.join(current_chunk + [sentence])) <= self.max_length:
                current_chunk.append(sentence)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        result = {i: chunk for i, chunk in enumerate(chunks) if len(chunk) > 0}

        with open('chunks.pkl', 'wb') as f:
            pickle.dump(result, f)

        return result

    def process_chunk(self, args):
        id, chunk = args
        atomic_facts, key_elements = self.extract_atomic_facts(chunk)
        return id, atomic_facts, key_elements

    def extract_atomic_facts(self, chunk):
        """Function to extract atomic facts and key elements."""
        # For simplicity, we use basic splitting; in practice, use an LLM or NLP library.
        atomic_facts = summarize_text(chunk, api_key=self.api_key)
        key_elements = set(generate_key_elements(
            atomic_facts, api_key=self.api_key).lower().split(","))
        key_elements = {element.strip() for element in key_elements}
        return atomic_facts, key_elements

    def build_graph(self):
        """Build the graph from the document."""
        chunks = self.chunk_document()

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #   # Map chunks to process_chunk function
        #   results = list(tqdm(executor.map(self.process_chunk, chunks.items()), total=len(chunks)))

        for id, chunk in tqdm(chunks.items()):
            atomic_facts, key_elements = self.extract_atomic_facts(chunk)
            for element in key_elements:
                if element not in self.key_elements:
                    self.key_elements[element] = [
                        {"facts": atomic_facts, "chunk_id": id}]
                else:
                    self.key_elements[element].append(
                        {"facts": atomic_facts, "chunk_id": id})

        # Create nodes and edges based on key elements
        for k, A in self.key_elements.items():
            self.graph.add_node(k, data=A)
            for other_k in self.key_elements:
                other_document = ". ".join(
                    [i["facts"] for i in self.key_elements[other_k]])
                if k != other_k and (k in other_document):
                    self.graph.add_edge(k, other_k)

    def get_graph(self):
        return self.graph

    def export_to_gml(self, file_path):
        """Export the graph to a GEXF file."""
        nx.write_gml(self.graph, file_path)
        print(f"Graph exported to {file_path}")

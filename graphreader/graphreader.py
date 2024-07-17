from pypdf import PdfReader
from tqdm import tqdm
from .document_graph import DocumentGraph
from .pinecone_client import push_to_pinecone
from .print_event import _print_event
from .prompts import get_primary_assistant_prompt
from langchain_openai.chat_models import ChatOpenAI
from .agent_graph import build_graph
import uuid
import os
from .tools import *


class GraphReader():

    def __init__(self,
                 document_path: str) -> None:

        print("Reading and chunking documents")
        reader = PdfReader(document_path)
        document = ""
        for page in tqdm(reader.pages):
            document += page.extract_text()

        doc_graph = DocumentGraph(document=document,
                                  max_length=1000)

        print("Building document graph")
        doc_graph.build_graph()
        graph = doc_graph.get_graph()

        doc_graph.export_to_gml("graph.gml")

        print("Pushing to Pinecone")
        push_to_pinecone(graph=graph)

    def get_response(self, query: str):
        primary_assistant_prompt = get_primary_assistant_prompt()
        tools = [search_tool, read_chunk,
                 stop_and_read_neighbor, get_node_data]
        llm = ChatOpenAI(model="gpt-3.5-turbo",
                         temperature=1, openai_api_key=os.getenv('gpt_api_key'))
        part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(
            tools)
        agent_graph = build_graph(part_1_assistant_runnable, tools)
        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {
                # Checkpoints are accessed by thread_id
                "thread_id": thread_id
            }
        }

        events = agent_graph.stream(
            {"messages": ("user", query)}, config, stream_mode="values"
        )
        # for event in events:
        #     pass

        # return event.get("messages")[-1].content
        _printed = set()
        for event in events:
            _print_event(event, _printed)

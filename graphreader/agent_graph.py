from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from .state import State
from .assistant import Assistant
from langgraph.prebuilt import ToolNode


def build_graph(part_1_assistant_runnable, tools):
    builder = StateGraph(State)

    # Define nodes: these do the work
    builder.add_node("assistant", Assistant(part_1_assistant_runnable))
    builder.add_node("tools", ToolNode(tools))
    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = SqliteSaver.from_conn_string(":memory:")
    part_1_graph = builder.compile(checkpointer=memory)
    return part_1_graph

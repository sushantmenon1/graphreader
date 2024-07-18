from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate


def summarize_text(text, api_key):
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a summarizing agent. Given a chunk of paragraph summarize it into atomic facts, the smallest indivisible facts that simplify the original text"},
            {"role": "user", "content": text}
        ]
    )

    return completion.choices[0].message.content


def generate_key_elements(text, api_key):
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a keyword extracting agent. Given a chunk of paragraph of a story extract only 3 key elements from it such as characters, places, and organizations. Return only the list of 3 words seperated by a comma and nothing else."},
            {"role": "user", "content": text}
        ]
    )

    return completion.choices[0].message.content


def get_primary_assistant_prompt():
    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            You are a helpful assistant which reads content from a graph and other information to assist the user's queries.
            Create a detailed plan on how to get the information from the graph.

            You have the following tools available:
            search_tool: Searches for the top 5 most relevant nodes based on the input query.
            get_node_data: Retrieves a list of dict with facts and their chunk ids. Use the nodes from the search_tool and read all the facts. Filter chunk_ids which might have the necessary information.
            read_chunk: Concatenates and returns the document text for the given chunk IDs. Any information you find useful use the write_to_notes function.
            stop_and_read_neighbor: Retrieve the facts and chunk_ids from neighboring nodes. Filter chunk_ids which might have the necessary information and then read the selected chunks using read_chunk function. Use this tool if you do not have enough information.
            write_to_notes: Any useful information you find, use this function to write it into the notes. Keep writing to this file as you find important information.
            read_notes: Use this function at the end to answer the users query.

            When searching, be persistent. Make sure to iterate through all the nodes in the result from the search_tool.
            Make sure to keep writing information to the notes.
            Never use the facts present in the results of the get_node_data function to directly answer the question. Make sure to always read from the notes.
            Do not make up any node not present in the result of the search_tool.
            """,
            ),
            ("placeholder", "{messages}"),
        ]
    )
    return primary_assistant_prompt

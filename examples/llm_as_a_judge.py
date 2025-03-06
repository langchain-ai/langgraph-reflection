from langgraph_reflexion import create_reflection_graph
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict

def call_model(state):
    return {"messages": init_chat_model(model="claude-3-7-sonnet-latest").invoke(state['messages'])}

graph = StateGraph(MessagesState).add_node(call_model).add_edge(START, "call_model").add_edge("call_model", END).compile()


class Finish(TypedDict):
    finish: bool



critique_prompt = """Your job is to look at the conversation below, and look at the AI assistant's response.

Critique it. Make sure it is complete. Make sure it is well thoughtout. Make sure it does what the user wants.

If it is good, then call the `Finish` tool. If you don't call the `Finish` tool, your response will be sent back to the assistant, so make sure to include concrete feedback on how it can improve."""

def nl_critique(state, config):
    response = init_chat_model(model="o3-mini", model_provider="openai").bind_tools([Finish]).invoke(
        [{"role": "system", "content": critique_prompt}] + state['messages']
    )
    if len(response.tool_calls) == 1:
        return
    else:
        return {"messages": [{"role": "user", "content": response.content}]}

critique_graph_general = StateGraph(MessagesState).add_node(nl_critique).add_edge(START, "nl_critique").add_edge("nl_critique", END).compile()

overall_graph = create_reflection_graph(graph, critique_graph_general)
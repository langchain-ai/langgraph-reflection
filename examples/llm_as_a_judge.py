from langgraph_reflexion import create_reflection_graph
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, Annotated, Literal
import json


# Define the main assistant model that will generate responses
def call_model(state):
    """Process the user query with a large language model."""
    model = init_chat_model(model="claude-3-7-sonnet-latest")
    return {"messages": model.invoke(state["messages"])}


# Define a basic graph for the main assistant
assistant_graph = (
    StateGraph(MessagesState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .add_edge("call_model", END)
    .compile()
)


# Define the tool that the judge can use to indicate the response is acceptable
class Finish(TypedDict):
    """Tool for the judge to indicate the response is acceptable."""

    finish: bool


# Define a more detailed critique prompt with specific evaluation criteria
critique_prompt = """You are an expert judge evaluating AI responses. Your task is to critique the AI assistant's latest response in the conversation below.

Evaluate the response based on these criteria:
1. Accuracy - Is the information correct and factual?
2. Completeness - Does it fully address the user's query?
3. Clarity - Is the explanation clear and well-structured?
4. Helpfulness - Does it provide actionable and useful information?
5. Safety - Does it avoid harmful or inappropriate content?

If the response meets ALL criteria satisfactorily, call the `Finish` tool to approve it.

If you find ANY issues with the response, do NOT call the Finish tool. Instead, provide specific and constructive feedback about what needs to be improved, and your response will be sent back to the assistant as a follow-up query.

Be detailed in your critique so the assistant can understand exactly how to improve."""


# Define the judge function with a more robust evaluation approach
def judge_response(state, config):
    """Evaluate the assistant's response using a separate judge model."""
    # Use a different model as the judge (can be smaller/more efficient)
    judge_model = init_chat_model(model="o3-mini", model_provider="openai").bind_tools(
        [Finish]
    )

    # Create judge prompt with all messages for context
    response = judge_model.invoke(
        [{"role": "system", "content": critique_prompt}] + state["messages"]
    )

    # If the judge called the Finish tool, the response is approved
    if len(response.tool_calls) == 1:
        print("✅ Response approved by judge")
        return
    else:
        # Otherwise, return the judge's critique as a new user message
        print("⚠️ Judge requested improvements")
        return {"messages": [{"role": "user", "content": response.content}]}


# Define the judge graph
judge_graph = (
    StateGraph(MessagesState)
    .add_node(judge_response)
    .add_edge(START, "judge_response")
    .add_edge("judge_response", END)
    .compile()
)


# Create the complete reflection graph
reflexion_app = create_reflection_graph(assistant_graph, judge_graph)
reflexion_app = reflexion_app.compile()


# Example usage
if __name__ == "__main__":
    # Example query that might need improvement
    example_query = [
        {
            "role": "user",
            "content": "Explain how nuclear fusion works and why it's important for clean energy",
        }
    ]

    # Process the query through the reflexion system
    print("Running example with reflexion...")
    result = reflexion_app.invoke({"messages": example_query})

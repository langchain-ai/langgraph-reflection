from typing import Optional, Type, Any, Literal
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage


def end_or_reflect(state) -> Literal[END, "graph"]:
    if len(state["messages"]) == 0:
        return END
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return "graph"
    else:
        return END


def create_reflection_graph(
    graph: CompiledStateGraph,
    reflection: CompiledStateGraph,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
) -> StateGraph:
    _state_schema = state_schema or graph.builder.schema
    rgraph = StateGraph(state_schema, config_schema=config_schema)
    rgraph.add_node("graph", graph)
    rgraph.add_node("reflection", reflection)
    rgraph.add_edge(START, "graph")
    rgraph.add_edge("graph", "reflection")
    rgraph.add_conditional_edges("reflection", end_or_reflect)
    return rgraph

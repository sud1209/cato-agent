from __future__ import annotations
from functools import partial
from langgraph.graph import StateGraph, START, END
from app.graph.state import CatoState
from app.graph.nodes.classifier import classify_intent
from app.graph.nodes.qualifier import qualify
from app.graph.nodes.objection import handle_objection
from app.graph.nodes.booking import book_appointment
from app.graph.nodes.info import handle_info


def _route_after_classify(state: CatoState) -> str:
    intent = state.get("intent", "general")
    if intent == "objection":
        return "handle_objection"
    if intent == "qualify":
        return "qualify"
    if intent == "book":
        return "book_appointment"
    # "info" and "general" both route to handle_info.
    # The info node handles graceful fallback for greetings when no context is retrieved.
    return "handle_info"


def _route_after_qualify(state: CatoState) -> str:
    result = state.get("qualification_result")
    if result == "qualified":
        return "book_appointment"
    return END  # unqualified or pending → END


def build_graph(retriever=None):
    """
    Build and compile the CatoState graph.
    `retriever` is an optional HybridRetriever injected into objection/info nodes.
    """
    builder = StateGraph(CatoState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("qualify", qualify)
    builder.add_node("handle_objection", partial(handle_objection, retriever=retriever))
    builder.add_node("book_appointment", book_appointment)
    builder.add_node("handle_info", partial(handle_info, retriever=retriever))

    builder.add_edge(START, "classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "handle_objection": "handle_objection",
            "qualify": "qualify",
            "book_appointment": "book_appointment",
            "handle_info": "handle_info",
        },
    )

    builder.add_conditional_edges(
        "qualify",
        _route_after_qualify,
        {
            "book_appointment": "book_appointment",
            END: END,
        },
    )

    builder.add_edge("handle_objection", END)
    builder.add_edge("book_appointment", END)
    builder.add_edge("handle_info", END)

    return builder.compile()


# Module-level compiled graph (used by main.py)
cato_graph = build_graph()

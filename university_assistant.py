"""Capstone: Intelligent University Assistant (LangGraph).

Classifier -> specialized agents -> tools -> final answer, with memory
(MemorySaver + thread_id), conditional routing and error handling.

Run:  python university_assistant.py
Requires: pip install -U langgraph langchain langchain-openai langchain-community
Set OPENROUTER_API_KEY in the environment (or you will be prompted).
"""
import ast
import operator as op
import os
from datetime import datetime
from getpass import getpass
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --------------------------------------------------------------------------
# LLM configuration (OpenRouter, same LangChain interface as OpenAI)
# --------------------------------------------------------------------------
def build_llm() -> ChatOpenAI:
    if not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = getpass("Enter your OpenRouter API key: ")
    return ChatOpenAI(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
    )

# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _safe_eval(node):
    """Evaluate arithmetic only (no names, calls or attribute access)."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression such as '25 * 4 + 10'. Supports + - * / ** % //."""
    try:
        return str(_safe_eval(ast.parse(expression.strip(), mode="eval")))
    except Exception:
        return "Invalid expression"


@tool
def current_datetime(_: str = "") -> str:
    """Return the current date and time. Use for any question about today's date or the current time."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")


def default_tools() -> list:
    tools = [calculator, current_datetime]
    try:  # optional extension: Wikipedia (pip install wikipedia)
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        tools.append(WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500)))
    except Exception:
        pass
    return tools

# --------------------------------------------------------------------------
# State, categories and prompts
# --------------------------------------------------------------------------
CATEGORIES = ["COURSES", "PROGRAMMING", "MATHEMATICS", "ADMINISTRATION", "GENERAL"]
NODE_NAMES = {c: f"{c.lower()}_agent" for c in CATEGORIES}


class AssistantState(TypedDict):
    messages: Annotated[list, add_messages]
    category: str


SYSTEM_PROMPTS = {
    "COURSES": "You are a university courses advisor. Explain course content, prerequisites, "
               "schedules and study strategies. If you lack institution-specific data, say so honestly.",
    "PROGRAMMING": "You are a programming tutor. Give clear explanations and correct Python code when appropriate.",
    "MATHEMATICS": "You are a mathematics tutor. Show steps. ALWAYS use the calculator tool for arithmetic.",
    "ADMINISTRATION": "You are a university administration assistant (enrollment, fees, certificates, procedures). "
                      "Never invent official rules or deadlines; recommend contacting the administration office when unsure.",
    "GENERAL": "You are a helpful university assistant. Use the date/time tool for questions about the current date or time.",
}

# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------
def build_assistant(llm, tools):
    llm_with_tools = llm.bind_tools(tools)

    def classify(state: AssistantState):
        last = state["messages"][-1]
        text = (last.content if hasattr(last, "content") else str(last)).strip()
        if not text:                                   # error: empty question
            return {"category": "EMPTY"}
        prompt = f"""
        Classify the student's question into exactly one category:
        COURSES (course content, syllabus, schedules, exams, grading),
        PROGRAMMING, MATHEMATICS,
        ADMINISTRATION (enrollment, fees, certificates, procedures, deadlines),
        GENERAL (anything else).

        Question: {text}

        Return only the category name.
        """
        try:
            raw = llm.invoke(prompt).content.strip().upper()
        except Exception as e:                         # error: API failure
            print("Classifier API error:", e)
            return {"category": "ERROR"}
        # error: unknown label -> GENERAL
        return {"category": next((c for c in CATEGORIES if c in raw), "GENERAL")}

    def make_agent(category: str):
        system = SystemMessage(content=SYSTEM_PROMPTS[category])

        def node(state: AssistantState):
            try:
                return {"messages": [llm_with_tools.invoke([system] + state["messages"])]}
            except Exception as e:                     # error: API failure
                return {"messages": [AIMessage(
                    content=f"Sorry, I ran into a technical problem while answering "
                            f"({type(e).__name__}). Please try again.")]}
        return node

    def fallback(state: AssistantState):
        if state["category"] == "EMPTY":
            msg = "Your question seems to be empty. Please type a question and I'll help."
        else:
            msg = "Sorry, I couldn't process your question right now. Please try again in a moment."
        return {"messages": [AIMessage(content=msg)]}

    def route_after_classifier(state: AssistantState):
        return NODE_NAMES.get(state["category"], "fallback")

    def route_after_agent(state: AssistantState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    def route_after_tools(state: AssistantState):
        return NODE_NAMES.get(state["category"], "general_agent")  # back to requesting agent

    b = StateGraph(AssistantState)
    b.add_node("classifier", classify)
    b.add_node("fallback", fallback)
    b.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    for c in CATEGORIES:
        b.add_node(NODE_NAMES[c], make_agent(c))

    b.add_edge(START, "classifier")
    b.add_conditional_edges(
        "classifier", route_after_classifier,
        {**{n: n for n in NODE_NAMES.values()}, "fallback": "fallback"},
    )
    for c in CATEGORIES:
        b.add_conditional_edges(NODE_NAMES[c], route_after_agent, {"tools": "tools", END: END})
    b.add_conditional_edges("tools", route_after_tools, {n: n for n in NODE_NAMES.values()})
    b.add_edge("fallback", END)
    return b.compile(checkpointer=MemorySaver())


def show_graph_mermaid(graph) -> str:
    """Mermaid source of the graph (paste into https://mermaid.live)."""
    return graph.get_graph().draw_mermaid()

# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
def ask(assistant, question: str, thread_id: str = "demo"):
    config = {"configurable": {"thread_id": thread_id}}
    result = assistant.invoke({"messages": [HumanMessage(content=question)]}, config)
    print("=" * 70)
    print("USER     :", question)
    print("CATEGORY :", result["category"])
    print("ASSISTANT:", result["messages"][-1].content)
    return result


def main():
    llm = build_llm()
    assistant = build_assistant(llm, default_tools())

    demo_queries = [
        "What topics are usually covered in a machine learning course?",  # COURSES
        "Write a Python function that computes the factorial of n.",      # PROGRAMMING
        "What is 125 * 38 + 17?",                                         # MATHEMATICS (calculator)
        "How do I request an enrollment certificate?",                    # ADMINISTRATION
        "What is today's date?",                                          # GENERAL (date/time tool)
        "   ",                                                            # error: empty question
    ]
    for q in demo_queries:
        ask(assistant, q)

    # Memory test: same thread_id
    ask(assistant, "My name is Ahmed.", thread_id="memory-test")
    ask(assistant, "What is my name?", thread_id="memory-test")

    # Full message trace including tool calls
    r = assistant.invoke(
        {"messages": [HumanMessage(content="Compute (15 + 27) * 3 using the calculator.")]},
        {"configurable": {"thread_id": "trace"}},
    )
    for m in r["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    main()

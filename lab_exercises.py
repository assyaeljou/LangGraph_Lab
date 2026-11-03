"""Lab exercises (Parts II-IX): state, LLM nodes, multi-node flow, routing,
tools, memory, visualization and a small routing evaluation.

Run:  python lab_exercises.py
Uses the LLM and tools defined in university_assistant.py.
"""
import time
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from university_assistant import build_llm, calculator, current_datetime


# ---- Exercise 2: state and a first workflow --------------------------------
class EchoState(TypedDict):
    message: str
    response: str


def exercise_2():
    def process_message(state: EchoState):
        return {"response": f"Agent received: {state['message']}"}

    b = StateGraph(EchoState)
    b.add_node("process", process_message)
    b.add_edge(START, "process")
    b.add_edge("process", END)
    print(b.compile().invoke({"message": "Hello LangGraph!", "response": ""}))


# ---- Exercise 4: an LLM node ----------------------------------------------
def exercise_4(llm):
    def chatbot(state: EchoState):
        return {"response": llm.invoke(state["message"]).content}

    b = StateGraph(EchoState)
    b.add_node("chatbot", chatbot)
    b.add_edge(START, "chatbot")
    b.add_edge("chatbot", END)
    out = b.compile().invoke({"message": "Explain artificial intelligence in three sentences.", "response": ""})
    print(out["response"])


# ---- Exercise 5: two-stage agent (analyzer -> answer) -----------------------
class QAState(TypedDict):
    question: str
    analysis: str
    answer: str


def exercise_5(llm):
    def analyze_question(state: QAState):
        prompt = f"""
        Analyze the following question.

        Question:
        {state['question']}

        Identify:
        1. The topic
        2. The user's intention
        3. The information needed
        """
        return {"analysis": llm.invoke(prompt).content}

    def generate_answer(state: QAState):
        prompt = f"""
        Answer the user's question.

        Question:
        {state['question']}

        Analysis:
        {state['analysis']}

        Give a clear and technically correct answer.
        """
        return {"answer": llm.invoke(prompt).content}

    b = StateGraph(QAState)
    b.add_node("analyzer", analyze_question)
    b.add_node("answer", generate_answer)
    b.add_edge(START, "analyzer")
    b.add_edge("analyzer", "answer")
    b.add_edge("answer", END)
    out = b.compile().invoke({"question": "What is reinforcement learning?", "analysis": "", "answer": ""})
    print(out["answer"])


# ---- Exercises 6-7 and Part VI: classifier + conditional routing -----------
class RouterState(TypedDict):
    question: str
    category: str
    answer: str


def build_router_graph(llm):
    def classify_question(state: RouterState):
        prompt = f"""
        Classify this question into exactly one category:

        - mathematics
        - programming
        - artificial_intelligence

        Question:
        {state['question']}

        Return only the category.
        """
        return {"category": llm.invoke(prompt).content.strip().lower()}

    def mathematics_agent(state):
        return {"answer": llm.invoke(f"Solve this mathematics question:\n\n{state['question']}").content}

    def programming_agent(state):
        return {"answer": llm.invoke(
            f"Answer this programming question.\nInclude Python code when appropriate.\n\n{state['question']}").content}

    def ai_agent(state):
        return {"answer": llm.invoke(
            f"Answer this artificial intelligence question\nat university level.\n\n{state['question']}").content}

    def route_question(state):
        category = state["category"]
        if "mathematics" in category:
            return "math"
        if "programming" in category:
            return "programming"
        return "ai"  # default / fallback branch

    b = StateGraph(RouterState)
    b.add_node("classifier", classify_question)
    b.add_node("math", mathematics_agent)
    b.add_node("programming", programming_agent)
    b.add_node("ai", ai_agent)
    b.add_edge(START, "classifier")
    b.add_conditional_edges("classifier", route_question,
                            {"math": "math", "programming": "programming", "ai": "ai"})
    for n in ("math", "programming", "ai"):
        b.add_edge(n, END)
    return b.compile()


def exercise_6_7(router_graph):
    for q in ["What is the derivative of x²?",
              "How do I create a REST API with Python?",
              "What is reinforcement learning?"]:
        r = router_graph.invoke({"question": q, "category": "", "answer": ""})
        print("=" * 60, "\nQUESTION:", q, "\nCATEGORY:", r["category"], "\nANSWER:", r["answer"])


# ---- Parts VII-VIII: tool-using agent -------------------------------------
def build_tool_graph(llm):
    tools = [calculator, current_datetime]
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    b = StateGraph(MessagesState)
    b.add_node("agent", agent)
    b.add_node("tools", ToolNode(tools))
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    b.add_edge("tools", "agent")                        # loop back with the result
    return b.compile()


def exercise_8(tool_graph):
    print(calculator.invoke("25 * 4 + 10"))             # 110
    print(calculator.invoke("__import__('os')"))        # Invalid expression
    print(current_datetime.invoke(""))
    for q in ["What is 125 * 38?", "What is the capital of Morocco?"]:
        r = tool_graph.invoke({"messages": [("user", q)]})
        print("=" * 60, "\nQ:", q)
        for m in r["messages"]:
            m.pretty_print()


# ---- Part IX: memory ------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def exercise_9(llm):
    def chatbot(state: ChatState):
        return {"messages": [llm.invoke(state["messages"])]}

    def make_builder():
        b = StateGraph(ChatState)
        b.add_node("chatbot", chatbot)
        b.add_edge(START, "chatbot")
        b.add_edge("chatbot", END)
        return b

    # 1) no persistence: the agent forgets
    g = make_builder().compile()
    g.invoke({"messages": [("user", "My name is Ahmed.")]})
    r = g.invoke({"messages": [("user", "What is my name?")]})
    print("WITHOUT memory:", r["messages"][-1].content)

    # 2) checkpointer + thread_id: the agent remembers within a thread
    mg = make_builder().compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "ahmed-1"}}
    print("Agent:", mg.invoke({"messages": [("user", "My name is Ahmed.")]}, cfg)["messages"][-1].content)
    print("Agent:", mg.invoke({"messages": [("user", "What is my name?")]}, cfg)["messages"][-1].content)
    other = {"configurable": {"thread_id": "someone-else"}}
    print("Other thread:", mg.invoke({"messages": [("user", "What is my name?")]}, other)["messages"][-1].content)


# ---- Visualization --------------------------------------------------------
def show_graph(g):
    try:
        from IPython.display import Image, display
        display(Image(g.get_graph().draw_mermaid_png()))
    except Exception as e:
        print("PNG rendering unavailable:", e)
        print(g.get_graph().draw_mermaid())


# ---- Evaluation harness ----------------------------------------------------
TEST_CASES = [
    ("What is the derivative of x²?", "mathematics"),
    ("Solve 2x + 3 = 11", "mathematics"),
    ("How do I reverse a list in Python?", "programming"),
    ("Write a function to check for palindromes.", "programming"),
    ("What is a convolutional neural network?", "artificial_intelligence"),
    ("Explain overfitting.", "artificial_intelligence"),
]


def evaluate_router(router_graph):
    correct, rows = 0, []
    for q, expected in TEST_CASES:
        t0 = time.time()
        got = router_graph.invoke({"question": q, "category": "", "answer": ""})["category"]
        ok = expected in got
        correct += ok
        rows.append((q, expected, got, ok, round(time.time() - t0, 2)))
    for row in rows:
        print(row)
    print(f"Routing accuracy: {correct}/{len(TEST_CASES)} = {correct / len(TEST_CASES):.0%}")


def main():
    llm = build_llm()
    exercise_2()
    exercise_4(llm)
    exercise_5(llm)
    router_graph = build_router_graph(llm)
    exercise_6_7(router_graph)
    tool_graph = build_tool_graph(llm)
    exercise_8(tool_graph)
    exercise_9(llm)
    show_graph(router_graph)
    show_graph(tool_graph)
    evaluate_router(router_graph)


if __name__ == "__main__":
    main()

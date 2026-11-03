# Building AI Agents with LangGraph

Practical lab for *Advanced Artificial Intelligence / Generative AI* (PhD level). It builds, step by step, from a one-node graph to an **Intelligent University Assistant** with a classifier, specialized agents, tools, memory, conditional routing and error handling.

**Stack:** Python 3.12 · LangGraph 1.2 · LangChain 1.4 · `langchain-openai` · OpenRouter (`openrouter/free`)

---

## Repository layout

```text
.
├── README.md                       # this file (architecture + how to run)
├── langgraph_lab_Statement.ipynb   # executed notebook with outputs and written answers
├── university_assistant.py         # capstone: tools, classifier, agents, graph, demo
├── lab_exercises.py                # Parts II-IX: state, routing, tools, memory, evaluation
└── requirements.txt
```

`lab_exercises.py` imports the LLM builder and tools from `university_assistant.py`.

---

## 1. System architecture (layers)

```mermaid
flowchart TB
    subgraph APP["Application layer"]
        NB["Notebook / scripts<br/>(lab_exercises.py, university_assistant.py)"]
    end

    subgraph LG["LangGraph: orchestration"]
        SG["StateGraph<br/>State, nodes, edges, conditional edges"]
        CP["Checkpointer: MemorySaver<br/>(thread_id = conversation memory)"]
        PB["Prebuilt: ToolNode, tools_condition"]
    end

    subgraph LC["LangChain: components"]
        CM["ChatOpenAI<br/>(common chat-model interface)"]
        MSG["Messages: Human, AI, System, Tool"]
        TL["@tool functions"]
    end

    subgraph EXT["External services"]
        OR["OpenRouter API<br/>model: openrouter/free"]
        WK["Wikipedia (optional tool)"]
    end

    NB --> SG
    SG --- CP
    SG --- PB
    SG --> CM
    SG --> TL
    CM --> MSG
    CM -->|HTTPS| OR
    TL --> WK
```

| Layer | Role |
|---|---|
| **LLM** (via OpenRouter) | Understands the request, reasons, and decides when a tool is needed. It does not execute actions itself. |
| **LangChain** | Provides the chat-model interface, message types and `@tool` definitions, independent of the provider. |
| **LangGraph** | Runs the workflow: shared state, nodes, edges, loops, conditional routing and persistence. |
| **Tools** | Python functions that perform actions: safe calculator, date/time, optional Wikipedia. |

---

## 2. Exercise graphs (Parts II-IX)

### 2.1 Echo workflow (Exercise 2) and LLM node (Exercise 4)

State: `{message, response}`. Each node returns a partial update that LangGraph merges into the state.

```mermaid
flowchart LR
    S((START)) --> P["process / chatbot<br/>reads message, writes response"] --> E((END))
```

### 2.2 Two-stage agent (Exercise 5)

State: `{question, analysis, answer}`.

```mermaid
flowchart LR
    S((START)) --> A["analyzer<br/>topic, intention, information needed"]
    A -->|analysis| G["answer<br/>generates final answer"]
    G --> E((END))
```

### 2.3 Router with conditional edges (Exercises 6-7, Part VI)

State: `{question, category, answer}`. `route_question` inspects `category` and returns the next node name; anything unrecognized falls back to `ai`.

```mermaid
flowchart LR
    S((START)) --> C["classifier<br/>mathematics / programming / artificial_intelligence"]
    C -->|"mathematics"| M["math agent"]
    C -->|"programming"| P["programming agent"]
    C -->|"otherwise (default)"| AI["ai agent"]
    M --> E((END))
    P --> E
    AI --> E
```

### 2.4 Tool-using agent (Parts VII-VIII)

State: `MessagesState`. The LLM is bound to the tools and decides whether to call one; the loop returns to the agent with the tool result.

```mermaid
flowchart LR
    S((START)) --> AG["agent<br/>LLM with bound tools"]
    AG -->|"tool call present<br/>(tools_condition)"| T["tools<br/>calculator, current_datetime"]
    T -->|"tool result"| AG
    AG -->|"no tool call"| E((END))
```

### 2.5 Memory (Part IX)

The `add_messages` reducer appends to `messages`. Persistence comes from the checkpointer and the `thread_id`.

```mermaid
flowchart LR
    subgraph T1["thread_id = ahmed-1"]
        direction LR
        U1["'My name is Ahmed.'"] --> G1["graph + MemorySaver"]
        U2["'What is my name?'"] --> G1
        G1 --> R2["'Your name is Ahmed.'"]
    end
    subgraph T2["thread_id = someone-else"]
        direction LR
        U3["'What is my name?'"] --> G2["graph + MemorySaver"]
        G2 --> R3["'I don't know your name.'"]
    end
```

---

## 3. Capstone: Intelligent University Assistant

### 3.1 Graph

```mermaid
flowchart TD
    U(["User question"]) --> S((START))
    S --> C["classifier<br/>COURSES / PROGRAMMING / MATHEMATICS<br/>ADMINISTRATION / GENERAL"]

    C -->|"EMPTY or ERROR"| F["fallback<br/>friendly error message"]

    subgraph AGENTS["Specialized agents (built by make_agent)"]
        direction LR
        A1["courses_agent"]
        A2["programming_agent"]
        A3["mathematics_agent"]
        A4["administration_agent"]
        A5["general_agent"]
    end

    C -->|"COURSES"| A1
    C -->|"PROGRAMMING"| A2
    C -->|"MATHEMATICS"| A3
    C -->|"ADMINISTRATION"| A4
    C -->|"GENERAL or unknown label"| A5

    AGENTS -.->|"tool call requested"| T["tools (ToolNode)<br/>calculator, current_datetime, Wikipedia"]
    T -->|"result returns to the SAME agent"| AGENTS
    AGENTS -->|"no tool call: final answer"| E((END))
    F --> E

    MEM[("MemorySaver<br/>checkpoint per thread_id")] -.- C
```

### 3.2 Shared state

```python
class AssistantState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation (memory)
    category: str                             # COURSES | ... | GENERAL | EMPTY | ERROR
```

### 3.3 Routing functions

| Function | Decides | Returns |
|---|---|---|
| `route_after_classifier` | Which agent handles the question | `<category>_agent`, or `fallback` for `EMPTY` / `ERROR` |
| `route_after_agent` | Does the agent need a tool? | `tools` if the last message has `tool_calls`, else `END` |
| `route_after_tools` | Who continues after the tool? | The agent matching `state["category"]` (default `general_agent`) |

### 3.4 Request lifecycle (math question with a tool)

```mermaid
sequenceDiagram
    actor User
    participant G as Graph (MemorySaver)
    participant C as classifier
    participant M as mathematics_agent
    participant L as LLM (OpenRouter)
    participant T as tools

    User->>G: "What is 125 * 38 + 17?" (thread_id)
    G->>C: state.messages
    C->>L: classification prompt
    L-->>C: "MATHEMATICS"
    C-->>G: category = MATHEMATICS
    G->>M: route_after_classifier
    M->>L: system prompt + messages
    L-->>M: AIMessage with tool_call calculator("125 * 38 + 17")
    G->>T: route_after_agent -> tools
    T-->>G: ToolMessage "4767"
    G->>M: route_after_tools (same agent)
    M->>L: messages + tool result
    L-->>M: final answer
    M-->>G: AIMessage, route_after_agent -> END
    G-->>User: "The result is 4767." (state checkpointed)
```

### 3.5 Error handling

```mermaid
flowchart TD
    Q["Incoming question"] --> EMP{"Empty text?"}
    EMP -->|"yes"| E1["category = EMPTY"] --> FB["fallback node:<br/>ask for a question"]
    EMP -->|"no"| CL{"Classifier call"}
    CL -->|"API exception"| E2["category = ERROR"] --> FB2["fallback node:<br/>try again later"]
    CL -->|"label returned"| LB{"Known label?"}
    LB -->|"no"| GEN["default to GENERAL"]
    LB -->|"yes"| AG["specialized agent"]
    GEN --> AG
    AG -->|"API exception"| MSG["polite apology message<br/>with exception type"]
    AG -->|"bad or failing tool call"| TE["ToolNode handle_tool_errors=True<br/>error returned to the agent"]
```

### 3.6 Requirements mapping

| # | Requirement | Implementation |
|---|---|---|
| 1 | Question classifier | `classify` node → 5 categories + `EMPTY` / `ERROR` |
| 2 | Specialized agents | `make_agent(category)` with one system prompt each |
| 3 | Memory | `MemorySaver` + `thread_id` |
| 4 | Tools | `calculator` (AST-based), `current_datetime`, optional Wikipedia |
| 5 | Conditional routing | `route_after_classifier`, `route_after_agent`, `route_after_tools` |
| 6 | Error handling | Empty input, unknown label, API errors, invalid tool calls |

---

## 4. Design decisions

- **Safe calculator.** The lab's `eval()` can run arbitrary Python. The implementation parses with `ast` and allows only arithmetic operators, so `__import__('os')` returns `Invalid expression`.
- **Same agent after a tool.** `route_after_tools` returns control to the agent that asked, so the specialist writes the final answer.
- **Provider-agnostic LLM.** OpenRouter is used through `ChatOpenAI` with a custom `base_url`; switching back to OpenAI only changes `build_llm()`.
- **Secrets.** The API key comes from `OPENROUTER_API_KEY` or `getpass`, never from source code.

## 5. Known limitations

- The classifier reads only the **last message**, so ambiguous follow-ups may be misrouted.
- All agents share **one tool set**; specialization comes from system prompts.
- `MemorySaver` is **in memory** and lost on restart; use a SQLite or Postgres checkpointer for persistence.
- `openrouter/free` is a routing alias, so results are **not fully reproducible** even at `temperature=0`.
- Routing evaluation uses 6 easy questions and a substring match; a larger set with exact matching and a confusion matrix is recommended.

## 6. Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key"                   # Windows: setx OPENROUTER_API_KEY "your-key"

python university_assistant.py      # capstone demo (6 queries, memory test, tool trace)
python lab_exercises.py             # Parts II-IX and routing evaluation
```

To print the Mermaid source of any compiled graph: `graph.get_graph().draw_mermaid()`.

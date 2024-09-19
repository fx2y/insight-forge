from typing import Optional, List, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from app.core.config import TEMPERATURE, MODEL_NAME, OPENAI_API_BASE, OPENAI_API_KEY
from app.core.self_discover.constant import SELECT_PROMPT, ADAPT_PROMPT, STRUCTURE_PROMPT, REASONING_PROMPT, \
    REASONING_MODULES

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()


# Define state type
class SelfDiscoverState(TypedDict):
    """State for the Self-Discover Agent."""
    reasoning_modules: str
    task_description: str
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_structure: Optional[str]
    answer: Optional[str]


# Load prompts
PROMPTS = {
    "select": ChatPromptTemplate.from_template(SELECT_PROMPT),
    "adapt": ChatPromptTemplate.from_template(ADAPT_PROMPT),
    "structure": ChatPromptTemplate.from_template(STRUCTURE_PROMPT),
    "reason": ChatPromptTemplate.from_template(REASONING_PROMPT),
}

# Define state keys
STATE_KEYS = {
    "select": "selected_modules",
    "adapt": "adapted_modules",
    "structure": "reasoning_structure",
    "reason": "answer",
}


def create_node(prompt_key: str):
    """Create a node for the graph."""

    def node_func(inputs: Dict) -> Dict:
        chain = PROMPTS[prompt_key] | ChatOpenAI(temperature=TEMPERATURE, model=MODEL_NAME,
                                                 openai_api_base=OPENAI_API_BASE,
                                                 openai_api_key=OPENAI_API_KEY) | StrOutputParser()
        return {STATE_KEYS[prompt_key]: chain.invoke(inputs)}

    return node_func


def create_graph() -> StateGraph:
    """Create the Self-Discover Agent graph."""
    graph = StateGraph(SelfDiscoverState)

    for node in ["select", "adapt", "structure", "reason"]:
        graph.add_node(node, create_node(node))

    graph.add_edge(START, "select")
    graph.add_edge("select", "adapt")
    graph.add_edge("adapt", "structure")
    graph.add_edge("structure", "reason")
    graph.add_edge("reason", END)

    return graph


def analyze_task(task_description: str, reasoning_modules: List[str]) -> Dict:
    """Analyze a task using the Self-Discover Agent."""
    graph = create_graph()
    app = graph.compile()

    result = app.invoke({
        "task_description": task_description,
        "reasoning_modules": "\n".join(reasoning_modules)
    }, config={"callbacks": [langfuse_handler]})

    return result


if __name__ == '__main__':
    # TASK_EXAMPLE = "Lisa has 10 apples. She gives 3 apples to her friend and then buys 5 more apples from the store. How many apples does Lisa have now?"
    # TASK_EXAMPLE = "Evaluate the impact of the Industrial Revolution on urban population growth in 19th century England."
    TASK_EXAMPLE = "Effectiveness of AI-driven content moderation systems on major social platforms"

    result = analyze_task(TASK_EXAMPLE, REASONING_MODULES)
    print(result)

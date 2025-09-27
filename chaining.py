"""to investigate chaining workflow in agents....where an llm is called does a task,another llm is then called to improved the
task then another and another forming a chain"""

from typing import TypedDict
from langgraph.graph import StateGraph,END,START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama

from dotenv import load_dotenv

load_dotenv()

llm = Ollama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434" 
)

class AgentState(TypedDict):
    joke:str
    topic:str
    improved_joke:str
    final_joke:str


def initial_joke(state:AgentState)->AgentState:
    """Generates iniatial joke based on a particular topic"""

    msg = llm.invoke(f"generate a joke that has not questions or exclamations, based on the following topic {state['topic']}")
    

    return {'joke': msg.content}

def punchline(state:AgentState)->AgentState:
    """Checks if a function is a punchline"""

    if '?' in state['joke'] or '!' in state['joke']:
        return "end"   
    return "continue"
    
def improved(state:AgentState)->AgentState:
    """Improves the joke by making it more funny"""

    msg = llm.invoke(f"Improve this joke to make it more funny and adding word play.Just have one improved which is short {state['joke']}")
    
    return {'improved_joke':msg.content}

def polish(state:AgentState)->AgentState:

    msg = llm.invoke(f"Improve and polish this joke by adding a twist on it and generally making it more funny.Ensure it is just one and make it short{state['improved_joke']}")

    return {"final_joke":msg.content}


workflow = StateGraph(AgentState)

workflow.add_node("initial",initial_joke)
workflow.add_node("improved",improved)
workflow.add_node("polish",polish)


workflow.add_edge(START,"initial")
workflow.add_conditional_edges(
    "initial",
    punchline,
    {
        "continue":"improved",
        "end":END 
    }
)


workflow.add_edge("improved","polish")
workflow.add_edge("polish",END)

chain = workflow.compile()

result = chain.invoke({"topic":"Quantum Mechanics"})

print(f"============initial joke===========\n")
print(f"{result['joke']}\n\n")

if "improved_joke" in result:
    print(f"============improved joke ============\n")
    print(f"{result['improved_joke']}\n\n")

    print(f"=============final joke =============\n")
    print(f"{result['final_joke']}\n")


print(f"\n THE END")
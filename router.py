from typing import TypedDict,Literal
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START,StateGraph,END
from pydantic import BaseModel,Field

load_dotenv()

class Router(BaseModel):
    step:Literal["joke","poem","story"] = Field(
        None,
        description="the next step in the routing process"
    )


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

router=llm.with_structured_output(Router)

class AgentState(TypedDict):
    input:str
    decision:str
    output:str


def joke_node(state:AgentState)->AgentState:
    """writes a joke"""

    result = llm.invoke(state['input'])
    return {'output':result}

def poem_node(state:AgentState)->AgentState:
    """writes a poem"""

    result = llm.invoke(state['input'])
    return {'output':result}


def story_node(state:AgentState)->AgentState:
    """writes a story"""

    result = llm.invoke(state['input'])
    return {'output':result}


def router_call(state:AgentState)->AgentState:
    """makes a decision"""

    decision = router.invoke(
        [
            SystemMessage(content="route the input to story ,poem or joke based on the user request"),
            HumanMessage(content=f"{state['input']}")
        ],
    )
    return {'decision':decision.step}

def route_decision(state:AgentState)->AgentState:
    
    if state['decision'] == "joke":
        return "joke"
    
    elif state['decision'] == "poem":
        return "poem"
    
    elif state['decision'] =="story":
        return "story"
    
    else:
        return "could not clearly identify where to route"
    

graph = StateGraph(AgentState)

graph.add_node("joke",joke_node)
graph.add_node("poem",poem_node)
graph.add_node("story",story_node)
graph.add_node("router",router_call)


graph.add_edge(START,"router")

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "joke":"joke",
        "poem":"poem",
        "story":"story"
    }

)


graph.add_edge("joke",END)

graph.add_edge("poem",END)

graph.add_edge("story",END)

app=graph.compile()

result = app.invoke({"input": "generate a beautiful poem on quantum mechanics"})
print(result['output'])
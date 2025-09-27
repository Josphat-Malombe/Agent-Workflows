from typing import TypedDict,Annotated,List
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
import operator
from langgraph.types import Send
from langchain_core.messages import HumanMessage,SystemMessage


load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

class Section(BaseModel):
    name: str = Field(
        description="Name for this sections of the report"
    )

    description:str =Field(
        description="Brief overview of main topics and concepts covered in this section"
    )

 
class Sections(BaseModel):
    sections: List[Section] = Field(
        description="sections of the report"
    )

planner=llm.with_structured_output(Sections)


class AgentState(TypedDict):
    topic:str
    sections:List[Section]
    completed_sections: Annotated[
        list,operator.add
    ]
    final_report:str

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]



def orchestrate(state:AgentState)->AgentState:
    """Generates plan for the report"""

    result = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report"),
            HumanMessage(content=f"Here is a topic for the report {[state['topic']]}")
        ]
    )

    return {"sections":result.sections}

def call_llm(state:WorkerState)->WorkerState:
    """worker writes a section of the report"""

    section = llm.invoke(
        [
            SystemMessage(content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."),
            HumanMessage(content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}")
                          
        ]
    )

    return {"completed_sections":[section.content]}

def synthesizer(state:AgentState)->AgentState:
    """Synthesizes the sections  into a full report"""

    completed_sections = state["completed_sections"]

    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report":completed_report_sections}


def assign_workers(state:AgentState):
    """assigns each worker to a section in the report"""

    return [Send("call_llm", {"section":s}) for s in state["sections"]]



graph = StateGraph(AgentState)

graph.add_node("orchestrator",orchestrate)
graph.add_node("call_llm",call_llm)
graph.add_node("synthesizer",synthesizer)

graph.add_edge(START,"orchestrator")

graph.add_conditional_edges(
    "orchestrator",
    assign_workers,
    ["call_llm"]
)

graph.add_edge("call_llm", "synthesizer")
graph.add_edge("synthesizer",END)

app=graph.compile()

result=app.invoke({"topic":"create report on theories of Quantum Mechanics"})

print(result["final_report"])
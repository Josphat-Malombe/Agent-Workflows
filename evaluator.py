from typing import TypedDict,Literal
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field


load_dotenv()


class AgentState(TypedDict):
    topic:str
    joke:str
    feedback:str
    funny_or_not:str



class Feedback(BaseModel):
    grade:Literal["funny","not funny"] = Field(description="grades whether the joke is funny or not")

    feedback:str=Field(description="provides a short description on how to improve the joke")


llm=ChatGoogleGenerativeAI(model='gemini-1.5-flash')

evaluator=llm.with_structured_output(Feedback)

 
def generator(state:AgentState)->AgentState:
   """Generates a joke"""

   if state.get('feedback'):
       result=llm.invoke(
      f"write a joke based on the topic: {state['topic']} and take into consideration feedback: {state['feedback']}"
   )
   else:
       result = llm.invoke(
      f"Generate a joke based on the topic: {state['topic']}"
   )

   return {"joke": result.content}




def evaluator_node(state:AgentState)->AgentState:
    """Grades a joke whether its funny or not"""

    result = evaluator.invoke(f"Grade the joke :{state['joke']}")

    return{"funny_or_not":result.grade, "feedback":result.feedback}



def take_route(state:AgentState)->AgentState:
    """routes to either llm or end depending on the evaluator output"""

    if state['funny_or_not']=="funny":
        return 'accept'
    
    elif state['funny_or_not']=="not funny":
        return 'reject'
    


graph = StateGraph(AgentState)

graph.add_node("generator",generator)
graph.add_node("eval",evaluator_node)

graph.add_edge(START,"generator")
graph.add_edge("generator","eval")

graph.add_conditional_edges(
    "eval",
    take_route,
    {
        "accept":END,
        "reject":"generator"
    }
)

graph.add_edge("eval" ,END)

app=graph.compile()

result = app.invoke({"topic":"Quantum Mechanics"})

print(result['joke'])
"""This is to investigate parallelization workflow where you different task are being performed silmultaneousily by calling llm severally
and later aggregating final results...there are two variations: sectioning-breaking task into smaller,voting-running same taask severally
to have a variation of the results"""


from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END


load_dotenv()


class AgentState(TypedDict):
    topic:str
    poem:str
    joke:str
    story:str
    combined_text:str

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def generate_joke(state:AgentState)->AgentState:
    """Uses provided topic to come up with a single funny joke"""

    output = llm.invoke(f"Generate a single joke that is very funny and not a long one based on the following topic {state['topic']}")

    return {"joke":output.content}

def create_poem(state:AgentState)->AgentState:
    """Generates a beautiful poem """

    output=llm.invoke(f"Generate a very beautiful poem of about 5 to 10 lines based on this topic: {state['topic']}")

    return {"poem":output.content}

def write_story(state:AgentState) ->AgentState:
    """writes a story"""

    output=llm.invoke(f"In not less than 100 words and not more than 150, generate a creative story based on the following topic: {state['topic']}")

    return {"story":output.content}

def aggregate(state:AgentState)->AgentState:
    """aggregates the poem,joke and story"""

    combined = f"Here is a joke, poem and story based on the topic {state['topic']}\n\n"
    combined+=f"JOKE\n {state['joke']}\n\n"
    combined+=f"POEM\n{state['poem']}\n\n"
    combined+=f"STORY\n{state['story']}\n\n"

    return {'combined_text':combined}


graph = StateGraph(AgentState)

graph.add_node("joke",generate_joke)
graph.add_node("poem",create_poem)
graph.add_node("story",write_story)
graph.add_node("combine",aggregate)

graph.add_edge(START,"joke")
graph.add_edge(START,"poem")
graph.add_edge(START,"story")

graph.add_edge("joke","combine")
graph.add_edge("poem","combine")
graph.add_edge("story","combine")

graph.add_edge("combine",END)

app=graph.compile()

result = app.invoke({"topic":"Software Development"})

print(result['combined_text'])
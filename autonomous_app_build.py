import os
from langchain_google_genai import ChatGoogleGenerativeAI
import operator
from typing import Annotated, List, Tuple, TypedDict
from typing import Literal
from langchain_core.output_parsers import StrOutputParser
import subprocess
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START,END
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import asyncio
import re
import streamlit as st
st.title("InstaDev")


#Adding Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

for idx,message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.code(message["content"][0], language='python')
            st.download_button(
                label="Download Code",
                data=message["content"][0],       # Data to be downloaded
                file_name="download_app.py",      # Name of the downloaded file
                mime="python",                    # File type
                key = str(idx)
            )
            st.download_button(
            label="Download Readme",
            data=message["content"][1],          # Data to be downloaded
            file_name="ReadMe.txt",              # Name of the downloaded file
            mime="text/plain",                   # File type
            key = 'r' + str(idx) 
            )
        else:
            st.code(message["content"], language='python')

# CONNECTING GEMINI

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import os
key_path='C:/Users/amogh.prabhu/Downloads/rich-tome-436705-f7-8244445b57d1.json'
credentials = Credentials.from_service_account_file(key_path)
PROJECT_ID = 'rich-tome-436705-f7'
REGION = 'us-central1'
import vertexai
vertexai.init(project = PROJECT_ID, location = REGION, credentials = credentials)
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    max_retries=6,
    stop=None,
)

tools = []

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a python code generator and you are a part big chain your only task is to generate code as per the user requirement at the end you shoud generate code for the given task."),
        ("system", "generate only code dont do any extra things and generate code between ``` code ```"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
code_agent_executor = create_react_agent(llm, tools, state_modifier=prompt)


doc_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a documentation creater for the code given by user carefully study and the code   and neatly create a professional documentation for the code"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
document_generator = create_react_agent(llm,tools, state_modifier=doc_prompt)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    test_status: str | int
    done : bool = False 

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: str = Field(description="single step instruction to build the complete app")

from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, generate a single step instruction that instructs the coder to write the entire code for the desired application. \
            The step should be in the form of a single string. Format it as follows: \
            "generate a code for <application description> using <module or framework> with extra features if required". \
            Make sure to specify the correct module or framework based on the type of application. For example:
            Follow this exact structure for all tasks Remember only one step instruction and dont give instructions for installation since modules are already installed."""
        ),
        ("user","For a calculator app:"),
        ("system",'''generate a code for complete calculator app using tkinter module'''),
        ("user","For a snake game app:"),
        ("system",'''generate a code for snake game using pygame module'''),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | llm.with_structured_output(Plan)  # LLM chain

replanner_prompt = ChatPromptTemplate.from_template("""
    Your objective was this:
    {input}

    And your previous plan was 
    {plan}

    And your previous code was 
    {past_steps}

    And Error message was
    {test_status}

    {done}
    considering the above case generate a single step like below
    - For a calculator app: "generate a code for complete calculator app using tkinter module"
    - For a snake game app: "generate a code for snake game using pygame module"
    Follow this exact structure for all tasks. If any module is not found make the step such that it should use subprocess to install the module"""
)

replanner = replanner_prompt | llm.with_structured_output(Plan)
parser = StrOutputParser()

class Dep_List(BaseModel):
    """Plan to follow in future"""
    modules: List[str] = Field(description="list of names(string format) of modules that need to be installed")

install_dep_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system","""Given a Python code snippet, extract all external dependencies that need to be installed via pip. Return a list of valid Python package names that are compatible with pip install. If the module is part of the Python standard library, exclude it from the list. Ensure that all names are in the correct format for pip, so each item in the list can be installed directly using pip install <module_name>.
            The response should be a Python list format with module names in string format.The modules names need nod be same so research ,think and output take care of _ and - and Cases"""),
            ("placeholder", "{messages}"),  
    ]
)
inst_dep_chain = install_dep_prompt | llm.with_structured_output(Dep_List)  # LLM chain

agent_response = ""

async def execute_step(state: PlanExecute):
    plan = state["plan"]
    task = plan[0]
    task_formatted = plan
    print("TASK FORMATTED",task_formatted)
    task_formatted = task_formatted + "generate only code dont do any extra things and generate code between ``` code ``` and dont generate extra things and dont use ``` except for code"
    global agent_response
    agent_response = await code_agent_executor.ainvoke({"messages": [("user", task_formatted)]})
    print("agent response : ",agent_response)
    text = agent_response["messages"][-1].content
    in_bet = re.findall(r"```python(.*?)```", text, re.DOTALL)[0]
    code_block = parser.parse(in_bet)
    return {
        "past_steps": [(task, code_block)],"done":False,
    }

async def install_dependencies(state:PlanExecute):
    text = agent_response["messages"][-1].content
    in_bet = re.findall(r"```python(.*?)```", text, re.DOTALL)[0]
    code_block = parser.parse(in_bet)
    p = inst_dep_chain.invoke({"messages": [("user",code_block)]})
    # print(p.modules)
    res = subprocess.run(['pip','list'],capture_output=True,text = True)
    # print("RES  ",res)
    res = res.stdout
    # if type(res) != 
    if not isinstance(p, list):
        return{"done":False}
    for module in p.modules:
        s1 = module
        s2 = module.replace('-','_')
        s3 = module.replace('_','-')
        if (s1 not in res) and (s2 not in res) and (s3 not in res):
            res = subprocess.run(['pip','install',module],capture_output=True,text = True)
            print(f"succesfully installed {module}")
        else:
            print(f"{module} already installed")
    return {"done":False}


async def test_code(state:PlanExecute):
    code_block = state["past_steps"][-1][1]
    try :
        with open('app.py', 'w') as f:
          f.write(code_block)
          f.close()
        if 'streamlit' in code_block:
            res = subprocess.run(['streamlit','run','app.py'],capture_output=True,text = True)
        else:    
            res = subprocess.run(['python' ,'app.py'],capture_output=True,text = True)
        if len(res.stderr) > 0:
            raise Exception(res.stderr)
        # st.session_state.messages.append({"role": "assistant", "content": code_block})
            # if st.button("END"):
            #     os._exit(0)
        return {"test_status": 1}
        
    except Exception as e:
      print("Exception Occured")
      return {"test_status": str(e)}

async def plan_step(state: PlanExecute):
    p = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": p.steps}


async def replan_step(state: PlanExecute):
    if state["test_status"] == 1:
        return {"done":True}
    output = await replanner.ainvoke(state)
    return {"plan": output.steps,"done":False,"test_status":0}

async def doc_gen(state: PlanExecute):
    text = agent_response["messages"][-1].content
    in_bet = re.findall(r"```python(.*?)```", text, re.DOTALL)[0]
    code_block = parser.parse(in_bet)
    doc_gen_response = await document_generator.ainvoke({"messages": [("user",code_block)]})
    documentation = doc_gen_response["messages"][-1].content
    # print("Documentation : ",doc_gen_response["messages"][-1].content)
    with st.chat_message("user"):
            st.markdown(st.session_state.messages[-1]["content"])   
    st.session_state.messages.append({"role": "assistant", "content": [code_block,documentation]})
    with st.chat_message("assistant"):
        st.code(code_block, language='python')
            # st.markdown(code_block)
        st.download_button(
            label="Download Code",
            data=code_block,          # Data to be downloaded
            file_name="download_app.py",      # Name of the downloaded file
            mime="python",           # File type
            key = "Down"
        )
        st.download_button(
            label="Download Readme",
            data=documentation,          # Data to be downloaded
            file_name="ReadMe",      # Name of the downloaded file
            mime="text/plain",           # File type
            key = "Doc"
        )
    return {"done":True}

def should_end(state: PlanExecute) -> Literal["agent", "document_gen"]:
    if state['done']:
        return "document_gen"
    return "agent"



workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("install",install_dependencies)
workflow.add_node("test_", test_code)
workflow.add_node("document_gen",doc_gen)
workflow.add_node("replan", replan_step)


workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent","install")
workflow.add_edge("install", "test_")
workflow.add_edge("test_", "replan")
workflow.add_edge("document_gen",END)
workflow.add_conditional_edges("replan",should_end)


app = workflow.compile()

config = {"recursion_limit": 50}

# inputs = {"input": "Build calculator app in python with gui with all functionalites and shows error when divided by zero"}

# expression = st.text_input("Enter The Input Format:")
# st.button("SUBMIT",key = "button")
# x = input("ENTER THE INPUT PROMPT : ")

async def process_events(inputs):
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)

if prompt := st.chat_input("Enter The Input Prompt : ?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    x = prompt
    inputs = {"input": x}
    asyncio.run(process_events(inputs))
    # with st.chat_message("user"):
    #     st.markdown(prompt)

        # print(event)
# if st.session_state.button == True:
#     asyncio.run(process_events())
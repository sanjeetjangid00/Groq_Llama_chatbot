from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(model="llama-3.3-70b-versatile")

class ChatbotState(TypedDict):
    message : Annotated[list[BaseMessage], add_messages]

def chatbot_llm(state:ChatbotState):
    response = llm.invoke(state['message'])
    return {"message":[AIMessage(content=response.content)]}

graph = StateGraph(ChatbotState)
graph.add_node("chatbot_llm", chatbot_llm)
graph.add_edge(START, "chatbot_llm")
graph.add_edge("chatbot_llm", END)

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)


if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Header Section
st.markdown(
    f'<p style="text-align: center; font-size: 36px; font-weight: bold; color: #1A73E8;">{"GROQ Llama Chatbot"}</p>',
    unsafe_allow_html=True
)
# Input Section
user_input = st.chat_input("Ask me anything...")
chat_history = []
for message in st.session_state['messages']:

    if message['role'] == 'user':
        st.markdown(
        f'<div style="display: flex; justify-content: flex-end; margin: 10px 0;">'
        f'<div style="background-color: #D3E4FF; border-radius: 15px; padding: 10px; max-width: 110%; display: flex; align-items: center;">'
        f'<img src="https://img.icons8.com/ios/452/user-male-circle.png" style="vertical-align: middle; width: 25px; height: 25px; margin-right: 10px;" />'
        f'{message["content"]}</div></div>',
        unsafe_allow_html=True
    )
        chat_history.append(HumanMessage(content = message['content']))
    else:
        chat_history.append(AIMessage(content = message['content']))
        st.markdown(
        f'<div style="text-align: left; background-color: #E0F7FA; border-radius: 15px; padding: 10px; max-width: 110%; margin: 10px 0; display: inline-block;">'
        f'<img src="https://img.icons8.com/ios/452/artificial-intelligence.png" style="vertical-align: middle; width: 25px; height: 25px; margin-right: 10px;" />'
        f'{message["content"]}</div>',
        unsafe_allow_html=True)
        
             
if user_input:
    st.session_state['messages'].append({'role':'user', "content":user_input})
    
    st.markdown(
        f'<div style="display: flex; justify-content: flex-end; margin: 10px 0;">'
        f'<div style="background-color: #D3E4FF; border-radius: 15px; padding: 10px; max-width: 110%; display: flex; align-items: center;">'
        f'<img src="https://img.icons8.com/ios/452/user-male-circle.png" style="vertical-align: middle; width: 25px; height: 25px; margin-right: 10px;" />'
        f'{user_input}</div></div>',
        unsafe_allow_html=True
    )


    config = {"configurable": {'thread_id':1}}
    response = chatbot.invoke({'message': [HumanMessage(content = chat_history)]}, config = config)
    ai_message = response['message'][-1].content
    st.session_state['messages'].append({'role':'assistant', "content":ai_message})
    st.markdown(
        f'<div style="text-align: left; background-color: #E0F7FA; border-radius: 15px; padding: 10px; max-width: 110%; margin: 10px 0; display: inline-block;">'
        f'<img src="https://img.icons8.com/ios/452/artificial-intelligence.png" style="vertical-align: middle; width: 25px; height: 25px; margin-right: 10px;" />'
        f'{ai_message}</div>',
        unsafe_allow_html=True
    )       

        










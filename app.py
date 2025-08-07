from langchain_groq import ChatGroq
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

model = ChatGroq(model="llama-3.3-70b-versatile")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Header Section
st.markdown(
    f'<p style="text-align: center; font-size: 36px; font-weight: bold; color: #1A73E8;">{"GROQ Llama Chatbot"}</p>',
    unsafe_allow_html=True
)
# Input Section
chat_input = st.chat_input("Ask me anything...")

if chat_input:
    st.session_state.messages.append({"role": "user", "message": chat_input})
    response = model.invoke(chat_input)
    st.session_state.messages.append({"role": "ai", "message": response.content})

# Message Display Section
for message in st.session_state.messages:
    with st.chat_message("user"):
        # Display user message on the right with human logo
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end; margin: 10px 0;">'
            f'<div style="background-color: #D3E4FF; border-radius: 15px; padding: 10px; max-width: 110%; display: flex; align-items: center;">'
            f'<img src="https://img.icons8.com/ios/452/user-male-circle.png" style="vertical-align: middle; width: 25px; height: 25px; margin-right: 10px;" />'
            f'{message["message"]}</div></div>',
            unsafe_allow_html=True
        )
    with st.chat_message('assistant'):
        # Display AI response on the left with AI logo
        st.markdown(
            f'<div style="text-align: left; background-color: #E0F7FA; border-radius: 15px; padding: 10px; max-width: 110%; margin: 10px 0; display: inline-block;">'
            f'<img src="https://img.icons8.com/ios/452/artificial-intelligence.png" style="vertical-align: middle; width: 25px; height: 25px; margin-right: 10px;" />'
            f'{message["message"]}</div>',
            unsafe_allow_html=True
        )



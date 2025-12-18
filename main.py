
import streamlit as st
from tempfile import NamedTemporaryFile

from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_classic.agents import initialize_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage

from tools import ImageCaptionTool, ObjectDetectionTool


##############
# initialise agent
##############
tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key = 'chat_history',
    k = 5,
    return_messages = True
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key = 'gsk-XXXXXXXXXXXXXXXXXXXXXXXXX' #Add your own api key
)


agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
)


# Set title
st.title('Please ask a question to the image')

# Set header
st.header("Please upload an image")


# upload FileE
file = st.file_uploader("", type = ["jpeg", "jpg", "png"])


if file:
    # display image
    st.image(file, width=700)


    # text input
    user_question = st.text_input("Ask a question about your image.")


    ##############
    # compute Agent reponse
    ##############
    with NamedTemporaryFile(delete = False, dir = '.') as fp:
        fp.write(file.getbuffer())
        image_path = fp.name

    response = agent.run("{}, this is the image parth: {}".format(user_question, image_path))

    # Write agent response
    if user_question and user_question != "":
        st.write(response)

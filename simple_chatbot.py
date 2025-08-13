import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

import os
from dotenv import load_dotenv, find_dotenv
_= load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatbot = ChatOpenAI(model="gpt-4o-mini")

from langchain_core.messages import HumanMessage

# messagesToTheChatbot = [
#     HumanMessage(content="i love failures"),
# ]
# response = chatbot.invoke(messagesToTheChatbot)
# print(response)

# this chatbot doesnt have memory.

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.chains import LLMChain

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables = ["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{context}")
    ]
)

chain = LLMChain(
    llm = chatbot,
    prompt = prompt,
    memory=memory
)
response = chain.invoke("what is my name")
print(response)




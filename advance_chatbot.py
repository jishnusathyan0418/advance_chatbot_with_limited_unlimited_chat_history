import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)


import os
from dotenv import load_dotenv, find_dotenv
_= load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatbot = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage


chatbotMemory = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]

chatbot_with_message_history = RunnableWithMessageHistory(
    chatbot,
    get_session_history,
)

session0 = {"configurable": {"session_id": "000"}}

# resFromChatbot = chatbot_with_message_history.invoke(
#     [HumanMessage(content="My favorite cricket player is virat Kohli")],
#     config=session0,
# )

# resFromChatbot = chatbot_with_message_history.invoke(
#     [HumanMessage(content="My favorite person is mom")],
#     config=session0,
# )
# resFromChatbot = chatbot_with_message_history.invoke(
#     [HumanMessage(content="who is my favorite person")],
#     config=session0,
# )
# print(resFromChatbot.content)

#CHATBOT WITH LIMITED MEMORY MESSAGE HISTORY


def limited_chat_messages(messages, no_of_messages_to_keep=2):
    return messages[-no_of_messages_to_keep:]


prompt = ChatPromptTemplate.from_messages(
    ["system",
    "you are a very helpfull chatbot so please answer to the following questions",
    MessagesPlaceholder(variable_name="messages"),
    ]
)

limitedMemory = (
    RunnablePassthrough.assign(messages= lambda x: limited_chat_messages(x["messages"]))
    | prompt
    | chatbot
)

chatbot_with_limited_memory_message_history = RunnableWithMessageHistory(
    limitedMemory,
    get_session_history,
    input_messages_key="messages",
)

session5 = {"configurable": {"session_id": "005"}}
responseFromChatbot = chatbot_with_limited_memory_message_history.invoke(
    {
        "messages": [HumanMessage(content="My favorite color is black")]
    },
    config=session5
)
responseFromChatbot = chatbot_with_limited_memory_message_history.invoke(
    {
        "messages": [HumanMessage(content="My favorite car is porsche")]
    },
    config=session5
)

responseFromChatbot = chatbot_with_limited_memory_message_history.invoke(
    {
        "messages": [HumanMessage(content="My hobby is going gym daily")]
    },
    config=session5
)

responseFromChatbot = chatbot_with_limited_memory_message_history.invoke(
    {
        "messages": [HumanMessage(content="what is my favorite car")]
    },
    config=session5
)
print(responseFromChatbot.content)
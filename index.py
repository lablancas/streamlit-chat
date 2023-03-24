from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
import pinecone
import openai
import streamlit as st
from streamlit_chat import message
from uuid import uuid4

index_name = st.secrets["PINECONE_INDEX_NAME"]
openai.organization = st.secrets["OPENAI_ORG_ID"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0)

# initialize pinecone
pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_ENVIRONMENT"])

index = pinecone.Index(index_name)
docsearch = Pinecone(index, embeddings.embed_query, "text")
chain = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True)

if 'message_history' not in st.session_state:
  st.session_state["message_history"] = []

def id():
  x = uuid4()
  return str(x)

def ask(question):
  if question == "":
    return

  answer = chain({ "query": question })
  print(answer["source_documents"])
  return answer["result"]

for m in st.session_state["message_history"]:
  message(m["s"],is_user=m["user"],key=m["key"]) # display all the previous message

def log(question):
  print("log", question)

placeholder = st.empty()
text = st.text_input("Enter a question:", key="input")

with placeholder.container():
  if (text != ""):
    print(f"Q: {text}")
    st.session_state["message_history"].append({"s": text, "user": True, "key": id()})
    message(text,is_user=True) # display all the previous message

    answer = ask(text)

    print(f"A: {answer}")
    st.session_state["message_history"].append({"s": answer, "user": False, "key": id()})
    message(answer,is_user=False) # display all the previous message

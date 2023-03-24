from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
import pinecone
import os
import sys
from dotenv import load_dotenv
import openai
import streamlit as st
from streamlit_chat import message
from uuid import uuid4

load_dotenv()

# exit if missing arguments
if len(sys.argv) < 2:
    print("Usage: python3 qa-langchain-streamlit.py <index_name>")
    sys.exit(1)

index_name = sys.argv[1]
openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
llm = OpenAI()

# initialize pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

index = pinecone.Index(index_name)
docsearch = Pinecone(index, embeddings.embed_query, "text")
chain = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch)

if 'message_history' not in st.session_state:
  st.session_state["message_history"] = []

def id():
  x = uuid4()
  return str(x)

def ask(question):
  if question == "":
    return

  answer = chain.run(question)
  return answer

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

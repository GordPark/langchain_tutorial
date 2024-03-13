from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=api_key)
# output = llm.invoke("2024년 주택 사업 정책에 대하여 알려줘")
# print(output)
prompt = ChatPromptTemplate.from_messages([
    ("system","너는 주택이 없는 사람을 행복하게 하기 위한 정부정책 안내 컨설턴트야"),
    ("user","{input}")
])
chain = prompt | llm
chain.invoke({"input":"2024년 주택 사업 정책에 대하여 알려줘"})
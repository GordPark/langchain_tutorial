from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 스타일 추천해주는 컨설턴트야"),
    ("user", "{input}")
])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

loader = WebBaseLoader("http://www.nature.go.kr/kbi/plant/clss/KBI_2001_010100.do")
docs = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key=api_key)

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>   # 주어진 대화 문맥을 나타냄
{context}   # 템플릿 내에서 변수
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

document_chain.invoke({
    "input": "육상식물은 뭐야?",
    "context": [Document(page_content="""식물의 정의
지구상의 생물계를 동물-식물-균류로 크게나누면 이들 중 세포벽이 있고 엽록소가 있어 독립영양으로 광합성을 하는 생물을 말한다. 또한 이동운동을 하지 않는 특징이 있다. """)]
})

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "육상식물이란 뭐야?"})
print(response["answer"])
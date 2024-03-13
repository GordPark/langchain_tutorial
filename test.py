# main.py
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime 
import time
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4-0125-preview"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container # 대화 내용을 표시할 streamlit 컨테이너
        self.text = initial_text    # 대화의 초기 텍스트

    # 새로운 토큰이 생성 될때마다 호출
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token  # 새로운 내용 계속 더하겠다, 대화 내용 지속적으로 업데이트
        self.container.markdown(self.text) # 새로운 내용을 마크다운 형식으로 컨테이너에 표시

want_to = """너는 전 세계 3월 축제에 대한 내용을 기반으로 질의응답을 하는 로봇이야.
content
{}
"""

content={}

def display_current_time():
    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.header("현재 시간: " + formatted)

st.header("전 세계 3월 축제")
st.info("매년 3월에 다양한 축제와 관련된 정보를 알아볼 수 있는 Q&A 로봇입니다.")
st.error("축제는 다양하고 원하시는 특정 지역이나 유형의 축제가 있으면 알려주시면 해당 정보를 더 제공해 드릴 수 있습니다.")

if "messages" not in st.session_state: 
    # 세션상태 초기화 assistant역할로 content를 세션상태 message에 추가
    st.session_state["messages"] = [ChatMessage(role="assistant", content="안녕하세요! 전 세계 3월 축제를 알려주는 Q&A 로봇입니다. 어떤 내용이 궁금하신가요?")]

for msg in st.session_state.messages:
    # role에 해당하는 역할의 채팅메세지를 streamlit 채팅 메세지 컴포넌트에 작성
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input(): # 사용자에게 채팅입력 요청, 입력 한다면
    # 사용자의 입력메세지를 user역할로 설정 후 세션 상태의 message에 추가
    # if "축제" in prompt:
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    # 사용자가 입력한 메세지를 streamlit 채팅 메세지 컴포넌트에 작성
    st.chat_message("user").write(prompt)

    if not API_KEY: # API_KEY가 비어있다면
        # 정보메세지로 문자열 메세지 표시
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    with st.chat_message("assistant"): # streamlit에서 assistant역할의 메세지 표시
        stream_handler = StreamHandler(st.empty()) # 비어 있는 컨테이너에 후속 메시지 표시
        # streaming스트리밍모드 사용여부 callbacks콜백 핸들러 목록
        llm = ChatOpenAI(openai_api_key=API_KEY, streaming=True, callbacks=[stream_handler], model_name=MODEL)
        # system 역할의 메세지 와 사용자로부터 받은 입력
        response = llm([ ChatMessage(role="system", content=want_to.format(content))]+st.session_state.messages)
        # reponse에서 받은 대화의 응답을 assistant 역할의 메세지로 변환 후 세션상태 메세지 목록에 추가
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))

st.sidebar.title("채팅목록")

# while True:
#     with     
#     display_current_time()
#     time.sleep(2)


import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_tavily import TavilySearch
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation, LLMResult
from typing import List
from pydantic import PrivateAttr
import requests

# --- Constants ---
NEVA_API_KEY = "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"
NEVA_URL = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
TAVILY_API_KEY = "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"

# --- Custom NVIDIA Neva LLM ---
class NevaLLM:
    def __init__(self, api_key=NEVA_API_KEY, url=NEVA_URL):
        self.api_key = api_key
        self.url = url

    def invoke(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.7,
            "seed": 0,
            "stream": False
        }
        response = requests.post(self.url, headers=headers, json=payload)
        if response.ok:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.text}"

# --- LangChain-Compatible Wrapper ---
class NevaLangChainWrapper(LLM):
    _neva_llm: NevaLLM = PrivateAttr()

    def __init__(self, neva_llm: NevaLLM):
        super().__init__()
        self._neva_llm = neva_llm

    def _call(self, prompt: str, **kwargs) -> str:
        return self._neva_llm.invoke(prompt)

    @property
    def _llm_type(self) -> str:
        return "neva-custom"

    def generate(self, prompts: List[str], **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            output = self._call(prompt)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)

def rephrase_output(text: str) -> str:
    prompt = f"Please rephrase this output in a more user-friendly and natural way for a better user experience:\n\n{text}\n\nRephrased:"
    return base_llm.invoke(prompt)
 
# --- Streamlit UI ---
st.set_page_config(page_title="LangChain + Neva-22B", page_icon="🤖")
st.title("🤖 LangChain + NVIDIA Neva-22B Demo")
st.sidebar.title("Choose a Demo")
mode = st.sidebar.radio("Demo", ["Basic Chat", "RAG (PDF Q&A)", "Agent"])

base_llm = NevaLLM()
wrapped_llm = NevaLangChainWrapper(base_llm)

# --- Basic Chat ---
if mode == "Basic Chat":
    st.header("💬 Basic Chat with Neva-22B")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

    user_input = st.text_input("You:", key="chat_input")
    if st.button("Send") and user_input:
        conversation = "".join([f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in st.session_state.chat_history])
        conversation += f"User: {user_input}\nAssistant:"
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = base_llm.invoke(conversation)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

# --- PDF Q&A ---
elif mode == "RAG (PDF Q&A)":
    st.header("📄 Ask Questions About a PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = splitter.split_documents(docs)

        def simple_retrieve(query: str, docs, k: int = 2):
            return sorted(docs, key=lambda d: query.lower() in d.page_content.lower(), reverse=True)[:k]

        question = st.text_input("Ask a question:")
        if st.button("Get Answer") and question:
            chunks = simple_retrieve(question, splits, k=2)
            context = "\n\n".join([chunk.page_content for chunk in chunks])
            prompt = f"Context:\n{context}\n\nUser: {question}\nAssistant:"
            with st.spinner("Answering..."):
                answer = base_llm.invoke(prompt)
            st.success(answer)

# --- Agent Mode: Weather Search ---
elif mode == "Agent":
    st.header("🌐 Ask Weather Info (via Tavily Search + Neva-22B)")

    search = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=3)

    def web_search_tool_func(query: str) -> str:
        return search.run(query)

    web_search_tool = Tool.from_function(
        func=web_search_tool_func,
        name="WebSearch",
        description="Search the web for up-to-date information like weather, news, etc."
    )

    agent_prefix = """You are a helpful assistant. Given a city name, use the WebSearch tool to get the current weather and provide a direct answer.

Your answer must be structured in this format:

City: <City Name>
Condition: <Weather description>
Temperature: <Temperature if available>

Use the tool to get the latest information, then output only the final answer in the format above.
"""

    agent = initialize_agent(
    tools=[web_search_tool],
    llm=wrapped_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    agent_kwargs={
        "prefix": agent_prefix,
        "output_parser_retries": 3  # ✅ Retry up to 3 times if parsing fails
    }
)

    user_input = st.text_input("Enter city name for weather (e.g., 'Panvel')", key="agent_input")
    if st.button("Get Weather") and user_input:
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke(user_input)
                #st.success(result["output"] if isinstance(result, dict) else result)
                friendly_output = rephrase_output(result)
                st.success(friendly_output)
            except Exception as e:
                st.error(f"Agent failed: {str(e)}")

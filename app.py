import json
import streamlit as st
import requests
from openai import OpenAI
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation, LLMResult
from typing import List
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
#from langchain.embeddings import OpenAIEmbeddings
from PIL import Image
from youtubesearchpython import VideosSearch
from googleapiclient.discovery import build
import datetime
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
from google.auth.transport.requests import Request
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch

def youtube_videos(query, max_results=5):
    YOUTUBE_API_KEY = "AIzaSyCmLgi49zKKdH3EZnal6cHkeXSByQPWGwc"  # <-- Replace with your actual API key
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    req = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results
    )
    res = req.execute()
    videos = []
    for item in res["items"]:
        videos.append({
            "title": item["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
            "thumbnail": item["snippet"]["thumbnails"]["default"]["url"]
        })
    return videos

def generate_search_query(user_input: str) -> str:
    """
    Use the LLM to generate a concise YouTube search query from user input.
    """
    prompt = (
        "You are an expert at generating concise YouTube search queries. "
        "Given a user's request, rewrite it as a short, effective search query for YouTube. "
        "Only return the search query, nothing else.\n\n"
        f"User request: {user_input}\nSearch query:"
    )
    # Use your LLM to generate the query
    query = base_llm.invoke(prompt)
    # Optionally, strip any extra whitespace or punctuation
    return query.strip()

def optimize_user_input(user_input: str) -> str:
    prompt = (
        "You are an expert prompt engineer. "
        "Rewrite the following user request to make it as clear, specific, and LLM-friendly as possible. "
        "Preserve the user's intent and style. Only return the improved prompt.\n\n"
        f"User request: {user_input}\nImproved prompt:"
    )
    improved = base_llm.invoke(prompt)
    return improved.strip()

def get_calendar_service():
    SCOPES = ['https://www.googleapis.com/auth/calendar.events']
    creds = None
    if os.path.exists('token_calendar.pickle'):
        with open('token_calendar.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token_calendar.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('calendar', 'v3', credentials=creds)
    return service

def add_event_to_calendar(summary, description, start_time, end_time):
    service = get_calendar_service()
    event = {
        'summary': summary,
        'description': description,
        'start': {
            'dateTime': start_time,
            'timeZone': 'Asia/Kolkata',  # Change to your timezone
        },
        'end': {
            'dateTime': end_time,
            'timeZone': 'Asia/Kolkata',
        },
    }
    event = service.events().insert(calendarId='primary', body=event).execute()
    return event.get('htmlLink')

def extract_event_details(user_input: str) -> dict:
    """
    Use the LLM to extract event details from a user's natural language request.
    Returns a dict with keys: summary, description, start_time, end_time (ISO format).
    """
    prompt = (
        "Extract the following details from the user's request to schedule a meeting: "
        "Event Title, Description, Start DateTime (ISO 8601), End DateTime (ISO 8601). "
        "If any detail is missing, make a reasonable guess. "
        "Return the result as a JSON object with keys: summary, description, start_time, end_time.\n\n"
        f"User request: {user_input}\nJSON:"
    )
    response = base_llm.invoke(prompt)
    try:
        event = json.loads(response)
        return event
    except Exception:
        return {}

# --- Constants ---
LLAMA_MODEL    = "nvidia/llama-3.1-nemotron-70b-instruct"
LLAMA_API_KEY  = "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"
LLAMA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# --- Custom Llama Client + LangChain Wrapper ---
class LlamaClient:
    def __init__(self, api_key=LLAMA_API_KEY, base_url=LLAMA_BASE_URL, model=LLAMA_MODEL):
        self.client    = OpenAI(base_url=base_url, api_key=api_key)
        self.model     = model

    def invoke(self, prompt: str, **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model       = self.model,
            messages    = [{"role": "user", "content": prompt}],
            temperature = kwargs.get("temperature", 0.5),
            top_p       = kwargs.get("top_p", 1.0),
            max_tokens  = kwargs.get("max_tokens", 1024),
            stream      = False
        )
        return resp.choices[0].message.content or ""

    def stream(self, prompt: str, **kwargs):
        """Yields token‐by‐token for a streaming UI."""
        # use the same payload but with stream=True
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.5),
            "top_p":       kwargs.get("top_p", 1.0),
            "max_tokens":  kwargs.get("max_tokens", 1024),
            "stream":      True
        }
        headers = {
            "Authorization": f"Bearer {LLAMA_API_KEY}",
            "Accept":        "application/json"
        }
        resp = requests.post(LLAMA_BASE_URL, json=payload, headers=headers, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta

class LlamaLangChainWrapper(LLM):
    _client: LlamaClient

    def __init__(self, client: LlamaClient):
        super().__init__()
        self._client = client

    def _call(self, prompt: str, **kwargs) -> str:
        return self._client.invoke(prompt, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "llama-nemotron"

    def generate(self, prompts: List[str], **kwargs) -> LLMResult:
        generations = []
        for p in prompts:
            text = self._call(p, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

# --- Replace in your Streamlit app ---
base_llm    = LlamaClient()
wrapped_llm = LlamaLangChainWrapper(base_llm)

# --- Chat Personalities ---
CHAT_MODES = {
    "Friendly": "You are a friendly and helpful assistant.",
    "Professional": "You are a professional and concise assistant.",
    "Creative": "You are a creative and imaginative assistant.",
    "Concise": "You are a concise and direct assistant."
}

def build_prompt(user_input: str, chat_mode: str, history: list) -> str:
    system_prompt = CHAT_MODES.get(chat_mode, CHAT_MODES["Friendly"])
    conversation = ""
    for msg in history[-5:]:
        conversation += f"{msg['role'].capitalize()}: {msg['content']}\n"
    conversation += f"User: {user_input}\nAssistant:"
    return f"{system_prompt}\n\n{conversation}"

# --- Streamlit UI ---
st.set_page_config(page_title="Llama Chat", page_icon="🦙")

# --- Add Mode Switcher ---
st.sidebar.title("Choose a Mode")
mode = st.sidebar.radio("Mode", ["Chat", "RAG (PDF & Image Q&A)","Schedule Meeting"], index=0)

# --- Chat Mode ---
if mode == "Chat":
    st.title("🦙 Llama Chatbot Demo")
    chat_mode = st.sidebar.selectbox("Chat Style", list(CHAT_MODES.keys()), index=0)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.chat_history:
        chat_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history])
        st.download_button("Download Chat", chat_text, file_name="chat_history.txt")

    for msg in st.session_state.chat_history:
        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

    user_input = st.text_input("You:", key="chat_input")
    if st.button("Send") and user_input:
        improved_input = optimize_user_input(user_input)
        prompt = build_prompt(user_input, chat_mode, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = base_llm.invoke(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
        
# --- Schedule Meeting Mode ---

elif mode == "Schedule Meeting":
    st.title("📅 Schedule a Meeting (NLP)")
    nlp_input = st.text_input("Describe your meeting (e.g., 'Team sync tomorrow at 3pm for 1 hour')")
    if st.button("Schedule from Description"):
        event = extract_event_details(nlp_input)
        if all(k in event for k in ("summary", "description", "start_time", "end_time")):
            try:
                link = add_event_to_calendar(
                    event["summary"], event["description"], event["start_time"], event["end_time"]
                )
                st.success(f"Event added! [View in Google Calendar]({link})")
            except Exception as e:
                st.error(f"Failed to add event: {e}")
        else:
            st.error("Could not extract all event details. Please try rephrasing your request.")

    # (Optional) Keep the manual form as a fallback
    with st.expander("Or fill in details manually"):
        with st.form("calendar_form"):
            summary = st.text_input("Event Title")
            description = st.text_area("Event Description")
            date = st.date_input("Event Date")
            start_time = st.time_input("Start Time")
            end_time = st.time_input("End Time")
            submitted = st.form_submit_button("Add to Google Calendar")
            if submitted:
                start_dt = datetime.datetime.combine(date, start_time).isoformat()
                end_dt = datetime.datetime.combine(date, end_time).isoformat()
                try:
                    link = add_event_to_calendar(summary, description, start_dt, end_dt)
                    st.success(f"Event added! [View in Google Calendar]({link})")
                except Exception as e:
                    st.error(f"Failed to add event: {e}")

# --- RAG Mode (PDF & Image Q&A) ---
elif mode == "RAG (PDF & Image Q&A)":
    st.title("📄🖼️ PDF & Image Q&A (RAG)")
    uploaded_pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    all_pdfs = []
    # --- Vectorstore creation for PDF ---
    if uploaded_pdfs:
        if "pdf_vectorstore" not in st.session_state or st.session_state.get("pdf_files") != [f.name for f in uploaded_pdfs]:
            progress_pdf = st.progress(0, text="Processing PDFs...")
            all_pdfs = []
            for idx, file in enumerate(uploaded_pdfs):
                with open(f"temp_{idx}.pdf", "wb") as f:
                    f.write(file.read())
                loader = PyPDFLoader(f"temp_{idx}.pdf")
                docs = loader.load()
                all_pdfs.extend(docs)
                progress_pdf.progress(int(100 * (idx + 1) / len(uploaded_pdfs)), text=f"Loaded {idx+1}/{len(uploaded_pdfs)} PDFs")
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = splitter.split_documents(all_pdfs)
            embeddings = HuggingFaceEmbeddings()
            vectordb = FAISS.from_documents(splits, embeddings)
            st.session_state.pdf_vectorstore = vectordb
            st.session_state.pdf_files = [f.name for f in uploaded_pdfs]
            progress_pdf.progress(100, text="PDFs processed!")
        pdf_vectorstore = st.session_state.pdf_vectorstore
        pdf_retriever = pdf_vectorstore.as_retriever()
    else:
        pdf_retriever = None
        st.session_state.pop("pdf_vectorstore", None)
        st.session_state.pop("pdf_files", None)

    # --- Vectorstore creation for Image ---
    if uploaded_image:
        progress_img = st.progress(0, text="Saving Image...")
        with open("temp_img.png", "wb") as f:
            f.write(uploaded_image.read())
        progress_img.progress(20, text="Loading Image...")
        loader = UnstructuredImageLoader("temp_img.png")
        docs = loader.load()
        progress_img.progress(40, text="Splitting image content...")
        splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        progress_img.progress(60, text="Generating embeddings...")
        embeddings = HuggingFaceEmbeddings()
        vectordb = FAISS.from_documents(splits, embeddings)
        progress_img.progress(100, text="Image processed!")
        image_vectorstore = vectordb
        image_retriever = image_vectorstore.as_retriever()
    else:
        image_retriever = None

    # --- Web Search Setup ---
    tavily = TavilySearch(api_key="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", max_results=3)

    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    question = st.text_input("Ask a question about your PDF or Image:", key="rag_input")
    if st.button("Get Answer"):
        context = ""
        improved_question = optimize_user_input(question)
        if pdf_retriever:
            docs = pdf_retriever.get_relevant_documents(question)
            context += "\n".join([doc.page_content for doc in docs])
        if image_retriever:
            docs = image_retriever.get_relevant_documents(question)
            context += "\n".join([doc.page_content for doc in docs])

        # --- Web Search ---
        web_results = tavily.run(question)
        web_context = "\n".join(web_results) if isinstance(web_results, list) else str(web_results)

        memory = "\n".join([f"User: {x['q']}\nAssistant: {x['a']}" for x in st.session_state.rag_history[-3:]])
        system_prompt = (
                "You are a helpful research assistant. "
                "Use the provided context from the user's PDF, image, and web search results to answer the user's question. "
                "Do not copy the context verbatim. Instead, synthesize, explain, or summarize the answer in your own words. "
                "If the answer is not in the context, say 'I couldn't find that information in the provided sources.' "
                "Cite both PDF/image and web sources if possible."
           )
        prompt = f"""{system_prompt}

Chat History:
{memory}

Context from PDF/Image:
{context}

Context from Web Search:
{web_context}

User Question:
{question}

Assistant:"""
        with st.spinner("Thinking..."):
            answer = base_llm.invoke(prompt)
        st.success(answer)
        st.session_state.rag_history.append({"q": question, "a": answer})

    # Show RAG chat history
    for msg in st.session_state.rag_history:
        st.markdown(f"**User:** {msg['q']}")
        st.markdown(f"**Assistant:** {msg['a']}")

# --- YouTube Video Search ---
st.header("🔎 Find Relevant YouTube Videos")
video_query = st.text_input("Enter a topic or question for YouTube search:", key="yt_query")
search_query = generate_search_query(video_query) if video_query else "latest news"  # Default search query if empty
if st.button("Search Videos"):
    with st.spinner("Searching YouTube..."):
        videos = youtube_videos(search_query, max_results=5)
    if videos:
        for vid in videos:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(vid["thumbnail"], width=120)
            with col2:
                st.markdown(f"[{vid['title']}]({vid['url']})")
    else:
        st.info("No videos found.")


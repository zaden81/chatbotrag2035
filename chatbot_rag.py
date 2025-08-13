# --- imports ---
import streamlit as st
st.set_page_config(page_title="Chatbot Pháp luật")

import os
from dotenv import load_dotenv
from math import gcd
from scipy.signal import resample_poly

# pinecone
from pinecone import Pinecone

# langchain + pinecone vector store
from langchain_pinecone import PineconeVectorStore

# retrievers & compressors
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever

# ONLY for ASR (giữ lại)
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

# embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# reranker
from sentence_transformers import CrossEncoder

# mic + LM Studio
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
from langchain_openai import ChatOpenAI

# audio utils
import io
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# --- init ---
load_dotenv()
# đảm bảo không chạy offline HF
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

st.title("Chatbot Tư vấn pháp luật")

# Options
with st.sidebar:
    st.header("Tùy chọn")
    fast_mode = st.toggle("⚡ Ưu tiên tốc độ", value=True, help="Tối ưu hoá độ trễ: tắt reranker/MultiQuery, giảm số token trả lời, ASR nhỏ hơn")

# Style to place mic button inline with chat input at the bottom
st.markdown(
    """
    <style>
      .mic-fab{position:fixed;bottom:20px;right:110px;z-index:1000;}
      .mic-fab > div{margin:0;}
      @media (max-width: 640px){.mic-fab{right:90px;bottom:16px;}}
    </style>
    """,
    unsafe_allow_html=True,
)
# Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index_host = os.environ.get("PINECONE_INDEX_HOST")

# Fallback if host is not provided
try:
    if index_host:
        index = pc.Index(index_name, host=index_host)
    else:
        # Try to fetch host from describe; if unavailable, instantiate without host
        desc = pc.describe_index(index_name)
        host = None
        if isinstance(desc, dict):
            host = desc.get("host")
        else:
            host = getattr(desc, "host", None)
        index = pc.Index(index_name, host=host) if host else pc.Index(index_name)
except Exception:
    # Last resort
    index = pc.Index(index_name)

# Embeddings 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Chat history
from langchain_core.messages import HumanMessage, AIMessage
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

@st.cache_resource
def load_lmstudio():
    client = OpenAI(
        base_url=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
    )
    model_id = os.getenv("LMSTUDIO_MODEL", "gemma-3-1b-it")

    llm_mq = ChatOpenAI(
        base_url=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        model=model_id,
        temperature=0.2,
        max_tokens=96,
    )
    return client, model_id, llm_mq


@st.cache_resource
def load_reranker():
    local_dir = r"D:\khoaluan\LangChain-Pinecone-RAG\documents\msmarco-minilm-l6-v2"  # Đường dẫn model đã tải
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        return CrossEncoder(local_dir, max_length=512, device=device)
    except Exception as e:
        st.error(f"Không load được model local: {e}")
        return None


# ==== PhoWhisper via transformers (pipeline ASR, không dùng ffmpeg) ====
@st.cache_resource
def load_asr_pipeline():
    """
    Load PhoWhisper-small từ thư mục local (offline).
    """
    import os
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # === ĐƯỜNG DẪN LOCAL CỦA MODEL (đổi cho đúng) ===
    LOCAL_MODEL_DIR = r"D:/khoaluan/LangChain-Pinecone-RAG/documents/phowhisper-small"

    try:
        processor = AutoProcessor.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            LOCAL_MODEL_DIR, local_files_only=True, torch_dtype=dtype
        )
        asr = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=getattr(processor, "tokenizer", processor),
            feature_extractor=getattr(processor, "feature_extractor", processor),
            device=device,
            chunk_length_s=20,
            stride_length_s=(4, 2),
            return_timestamps=False,
        )
        return asr
    except Exception as e:
        st.error(f"Không load được PhoWhisper-small từ local: {e}")
        return None

# Tải tài nguyên song song
with st.spinner("Đang tải mô hình và tài nguyên…"):
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_lms = executor.submit(load_lmstudio)
        future_reranker = executor.submit(load_reranker)
        future_asr = executor.submit(load_asr_pipeline)

        client, lm_model_id, llm = future_lms.result()
        reranker = future_reranker.result()
        asr_pipe = future_asr.result()



# --- Tạo retriever 3 tầng ---
# 1) Base: MMR để vừa liên quan vừa đa dạng
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "fetch_k": 60, "lambda_mult": 0.6},  
)

# 2) Nén/nội dung: lọc theo ngưỡng similarity để đuổi nhiễu
compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.50, top_k=10)
compressed_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=mmr_retriever)


# 3) MultiQuery: chỉ bật khi không phải fast mode
if fast_mode:
    mq_retriever = compressed_retriever
else:
    mq_retriever = MultiQueryRetriever.from_llm(retriever=compressed_retriever, llm=llm)


# --- Helper: Rerank + guardrail ---
def truncate_for_rerank(text: str, max_chars: int = 2500) -> str:
    return text if len(text) <= max_chars else text[:max_chars]

def retrieve_docs(query: str, final_k: int = 10):
    candidates = mq_retriever.get_relevant_documents(query)

    if not candidates and not fast_mode:
        old = compressor.similarity_threshold
        compressor.similarity_threshold = 0.5
        candidates = mq_retriever.get_relevant_documents(query)
        compressor.similarity_threshold = old

    if not candidates:
        simple = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6 if fast_mode else 8})
        candidates = simple.get_relevant_documents(query)

    if not candidates:
        return []

    use_reranker = (reranker is not None)

    if not use_reranker:
        ranked_docs = candidates[: (5 if fast_mode else final_k)]
    else:
        pairs = [(query, truncate_for_rerank(d.page_content)) for d in candidates]
        scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            
    if reranker is None:
        ranked_docs = candidates[: (5 if fast_mode else final_k)]
    else:
        pairs = [(query, truncate_for_rerank(d.page_content)) for d in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        seen, ranked_docs = set(), []
        for d, s in ranked:
            key = (d.metadata.get("source_id") or d.metadata.get("id") or d.metadata.get("name"),
                   d.metadata.get("chunk_idx"))
            if key in seen:
                continue
            seen.add(key)
            ranked_docs.append(d)
            if len(ranked_docs) >= final_k:
                break
    return ranked_docs

# --- System prompt ---
SYSTEM_PROMPT_TMPL = """Bạn là trợ lý pháp lý. Chỉ sử dụng thông tin trong CONTEXT để trả lời ngắn gọn, nêu rõ Điều/Khoản/Tên luật nếu xuất hiện. 
Nếu không tìm thấy trong CONTEXT thì nói “Tôi không có đủ thông tin để chắc chắn” và đề xuất câu hỏi làm rõ. 
"

Context:
{context}
"""

def build_context(docs):
    blocks = []
    for doc in docs:
        name = doc.metadata.get("name") or doc.metadata.get("file_name") or doc.metadata.get("source_id") or "Không rõ nguồn"
        blocks.append(f"[Nguồn: {name}]\n{doc.page_content}")
    return "\n\n".join(blocks)

# ==== Mic button (floating & visible above chat_input) ====
prompt = st.chat_input("Hãy đặt câu hỏi về Luật Hôn Nhân và Gia Đình")
with st.container():
  st.markdown('<div class="mic-fab">', unsafe_allow_html=True)
  audio_dict = mic_recorder(
      key="main_mic",
      start_prompt="🎙️ Bắt đầu ghi âm",
      stop_prompt="⏹️ Dừng ghi âm",
      just_once=False,   # bấm để ghi, bấm lần nữa để dừng
      format="wav"       # giữ 'wav' để đọc bằng soundfile
  )
  st.markdown('</div>', unsafe_allow_html=True)

# Nếu vừa dừng ghi âm → nhận dạng (đọc WAV từ bytes, tự resample) → đổ prompt nếu ô chat trống
voice_text = None
if audio_dict and "bytes" in audio_dict and audio_dict["bytes"] and asr_pipe:
    with st.spinner("Đang nhận dạng giọng nói…"):
        audio_bytes = audio_dict["bytes"]
        # Đọc WAV từ bytes (không cần ffmpeg)
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if data.ndim > 1:  # stereo -> mono
            data = np.mean(data, axis=1)

        # Lấy sample rate mong muốn của model (thường 16000)
        try:
            target_sr = asr_pipe.feature_extractor.sampling_rate
        except Exception:
            target_sr = 16000

        # Nếu khác SR, tự resample bằng polyphase (khỏi cần torchaudio)
        if sr != target_sr:
            g = gcd(int(sr), int(target_sr))
            up = int(target_sr) // g
            down = int(sr) // g
            data = resample_poly(data, up, down).astype("float32")
            sr = target_sr

        input_audio = {"array": data, "sampling_rate": sr}

        # Gọi ASR (ép tiếng Việt, không dịch)
        result = asr_pipe(input_audio, generate_kwargs={"language": "vi", "task": "transcribe"})
        voice_text = (result.get("text", "") or "").strip()
        if voice_text and not prompt:
            prompt = voice_text
            st.toast("🎙️ Đã lấy văn bản từ giọng nói", icon="✅")


# --- Run ---
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    docs = retrieve_docs(prompt, final_k=4)

    if not docs:
        safe_reply = "Tôi không có đủ thông tin để chắc chắn."
        with st.chat_message("assistant"):
            st.markdown(safe_reply)
        st.session_state.messages.append(AIMessage(safe_reply))
    else:
        docs_text = build_context(docs)
        system_prompt = SYSTEM_PROMPT_TMPL.format(context=docs_text)

        # Gọi model qua LM Studio (OpenAI-compatible), stream từng chunk
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            partial_text = ""

            # khi gọi LM Studio:
            stream = client.chat.completions.create(
                model=lm_model_id,
                messages=messages,
                temperature=0.0 if fast_mode else 0.7,
                max_tokens=200 if fast_mode else 300,  # ↓
                stream=True,
)

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    partial_text += delta
                    placeholder.markdown(partial_text)

        result = partial_text.strip()
        st.session_state.messages.append(AIMessage(result))

        with st.expander("Nguồn tham khảo đã dùng"):
            for i, d in enumerate(docs, 1):
                name = (d.metadata.get("name")
                        or d.metadata.get("file_name")
                        or d.metadata.get("source_id")
                        or "Không rõ nguồn")
                st.markdown(f"**{i}. {name}**")


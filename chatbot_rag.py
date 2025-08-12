# --- imports ---
import streamlit as st
st.set_page_config(page_title="Chatbot Pháp luật")  # phải là lệnh Streamlit đầu tiên

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

# huggingface & wrappers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# reranker (cross-encoder)
from sentence_transformers import CrossEncoder

# mic
from streamlit_mic_recorder import mic_recorder

# audio utils (để đọc WAV từ bytes, không cần ffmpeg)
import io
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# --- init ---
load_dotenv()
# đảm bảo không chạy offline HF
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

st.title("Chatbot Tư vấn pháp luật")

# Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index_host = os.environ.get("PINECONE_INDEX_HOST")
index = pc.Index(index_name, host=index_host)  # nếu cần host: pc.Index(index_name, host=index_host)

# Embeddings (All-MiniLM-L6-v2 là ổn cho tốc/chi phí)
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

# --- LLM & reranker caches ---
@st.cache_resource
def load_gemma():
    model_id = "google/gemma-3-1b-it"
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    # Tạo pipeline để dùng cho MultiQueryRetriever (ít random để ổn định câu hỏi phụ)
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=gen_pipe)
    return tokenizer, model, llm

@st.cache_resource
def load_reranker():
    """Ưu tiên model nhanh; nếu lỗi, fallback sang đa ngôn ngữ; nếu vẫn lỗi → None."""
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", token=hf_token, max_length=512)
    except Exception as e1:
        st.warning(f"Không tải được ms-marco-MiniLM-L-6-v2. Thử reranker đa ngôn ngữ…\n{e1}")
        try:
            return CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", token=hf_token, max_length=512, trust_remote_code=True)
        except Exception as e2:
            st.error(f"Không tải được reranker (đã thử 2 model). Chạy không rerank.\n{e2}")
            return None

# ==== PhoWhisper via transformers (pipeline ASR, không dùng ffmpeg) ====
@st.cache_resource
def load_asr_pipeline():
    """
    Tải PhoWhisper/Whisper theo thứ tự ưu tiên.
    Không dùng use_auth_token/local_files_only trong pipeline.
    Đọc WAV từ bytes bằng soundfile → không cần ffmpeg.
    """
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    repos = [
        "vinai/PhoWhisper-small",  # ưu tiên PhoWhisper (TV)
        "openai/whisper-small",    # fallback 1
        "openai/whisper-tiny"      # fallback 2 (nhẹ để test mạng/token)
    ]

    last_err = None
    for repo in repos:
        try:
            processor = AutoProcessor.from_pretrained(repo, token=hf_token, local_files_only=False)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                repo, token=hf_token, local_files_only=False, torch_dtype=dtype
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
            if repo != "vinai/PhoWhisper-small":
                st.info(f"Đang dùng fallback: {repo}")
            return asr
        except Exception as e:
            st.warning(f"Không tải được {repo}: {e}")
            last_err = e

    st.error(f"ASR chưa sẵn sàng. Kiểm tra HUGGINGFACE_HUB_TOKEN và kết nối mạng. Chi tiết: {last_err}")
    return None

# Tải các tài nguyên nặng song song để giảm thời gian chờ
with st.spinner("Đang tải mô hình và tài nguyên…"):
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_gemma = executor.submit(load_gemma)
        future_reranker = executor.submit(load_reranker)
        future_asr = executor.submit(load_asr_pipeline)
        tokenizer, model, llm = future_gemma.result()
        reranker = future_reranker.result()
        asr_pipe = future_asr.result()

# --- Tạo retriever 3 tầng ---
# 1) Base: MMR để vừa liên quan vừa đa dạng
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 12, "fetch_k": 60, "lambda_mult": 0.6},
)

# 2) Nén/nội dung: lọc theo ngưỡng similarity để đuổi nhiễu
compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.62)
compressed_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=mmr_retriever)

# 3) MultiQuery: sinh biến thể câu hỏi tăng recall
mq_retriever = MultiQueryRetriever.from_llm(retriever=compressed_retriever, llm=llm)

# --- Helper: Rerank + guardrail ---
def truncate_for_rerank(text: str, max_chars: int = 2500) -> str:
    return text if len(text) <= max_chars else text[:max_chars]

def retrieve_docs(query: str, final_k: int = 6):
    candidates = mq_retriever.get_relevant_documents(query)

    if not candidates:
        old = compressor.similarity_threshold
        compressor.similarity_threshold = 0.58
        candidates = mq_retriever.get_relevant_documents(query)
        compressor.similarity_threshold = old

    if not candidates:
        simple = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        candidates = simple.get_relevant_documents(query)

    if not candidates:
        return []

    if reranker is None:
        ranked_docs = candidates[:final_k]
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
SYSTEM_PROMPT_TMPL = """Bạn là trợ lý pháp lý, chỉ trả lời dựa trên dữ liệu được cung cấp.

YÊU CẦU:
- Trả lời đầy đủ ý chính.
- Ghi rõ Nguồn (trong ngữ liệu) và trích dẫn Điều/Khoản/Điểm/Tên luật (nếu xuất hiện trong ngữ liệu).
- KHÔNG bịa, KHÔNG suy đoán ngoài dữ liệu.
- Nếu không đủ thông tin: "Tôi không có đủ thông tin để chắc chắn."

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

    # Lấy tài liệu tốt nhất theo pipeline 3 tầng + (rerank nếu có)
    docs = retrieve_docs(prompt, final_k=6)

    # Nếu vẫn trống, trả lời an toàn
    if not docs:
        safe_reply = "Tôi không có đủ thông tin để chắc chắn."
        with st.chat_message("assistant"):
            st.markdown(safe_reply)
        st.session_state.messages.append(AIMessage(safe_reply))
    else:
        docs_text = build_context(docs)
        system_prompt = SYSTEM_PROMPT_TMPL.format(context=docs_text)

        # Gọi Gemma để trả lời
        final_prompt = system_prompt + "\n\nQuestion: " + prompt + "\nAnswer:"
        inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.7
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        result = decoded.split("Answer:")[-1].strip()

        with st.chat_message("assistant"):
            st.markdown(result)
        st.session_state.messages.append(AIMessage(result))

        # Hiển thị nguồn
        with st.expander("Nguồn tham khảo đã dùng"):
            for i, d in enumerate(docs, 1):
                name = (
                    d.metadata.get("name")
                    or d.metadata.get("file_name")
                    or d.metadata.get("source_id")
                    or "Không rõ nguồn"
                )
                st.markdown(f"**{i}. {name}**")

# Nhắc nếu ASR chưa sẵn sàng
if not asr_pipe:
    st.caption("⚠️ ASR chưa sẵn sàng (PhoWhisper/Whisper). Kiểm tra HUGGINGFACE_HUB_TOKEN và kết nối mạng.")

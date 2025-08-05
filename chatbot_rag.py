#import streamlit
import streamlit as st
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

st.set_page_config(page_title="Chatbot Pháp luật")
st.title("Chatbot Tư vấn pháp luật")

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# initialize pinecone database
index_name = os.environ.get("PINECONE_INDEX_NAME")  # change if desired
index = pc.Index(index_name)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("Hãy đặt câu hỏi về Luật Hôn Nhân và Gia Đình")

@st.cache_resource
def load_gemma():
    model_id = "google/gemma-3-1b-it"
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return tokenizer, model

tokenizer, model = load_gemma()

# did the user submit a prompt?
if prompt:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)

        st.session_state.messages.append(HumanMessage(prompt))


    # creating and invoking the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    docs = retriever.invoke(prompt)

    # Gộp các tài liệu đã truy xuất cùng metadata của chúng
    context_chunks = []
    for doc in docs:
        name = doc.metadata.get("name")
        context = f"[Nguồn: {name}]\n{doc.page_content}"
        context_chunks.append(context)

    docs_text = "\n\n".join(context_chunks)


    # creating the system prompt
    system_prompt = """Bạn là trợ lý pháp lý, chỉ trả lời dựa trên dữ liệu được cung cấp.

    Dựa vào các văn bản sau, hãy trả lời **đầy đủ ý chính**, kèm theo **Nguồn** và **trích dẫn rõ ràng điều, khoản, điểm, tên luật** nếu có. **Không được bịa thêm nội dung** hoặc đưa ra suy đoán ngoài dữ liệu.

    Nếu không thể xác định câu trả lời từ dữ liệu, hãy trả lời: "Tôi không có đủ thông tin để chắc chắn."

    Context: {context}
    """


    # Populate the system prompt with the retrieved context
    system_prompt_fmt = system_prompt.format(context=docs_text)


    print("-- SYS PROMPT --")
    print(system_prompt_fmt)

    # adding the system prompt to the message history
    st.session_state.messages.append(SystemMessage(system_prompt_fmt))

    # invoking the llm
    # Tạo input cho Gemma từ system prompt + user prompt
    final_prompt = system_prompt_fmt + "\n\nQuestion: " + prompt + "\nAnswer:"

    inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    result = decoded.split("Answer:")[-1].strip()


    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(result)

        st.session_state.messages.append(AIMessage(result))
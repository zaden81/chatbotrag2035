# import basics
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Dùng HuggingFaceEmbeddings miễn phí (same model with ingestion)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Tạo vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Tạo retriever
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},
)

# Query mẫu (bạn có thể thay câu hỏi tùy ý)
query = "Điều kiện đăng ký kết hôn là gì??"
results = retriever.invoke(query)

# In kết quả
print("\n=== KẾT QUẢ TRUY VẤN ===\n")
if not results:
    print("❌ Không tìm thấy kết quả phù hợp.")
else:
    for i, res in enumerate(results, 1):
        print(f"[{i}] {res.page_content}\n→ Metadata: {res.metadata}\n")

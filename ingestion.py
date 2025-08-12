# import basics
import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

#documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


load_dotenv()


pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")  # change if desired

# check whether index exists, and create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# initialize embeddings model + vector store

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# loading the PDF document
import json

json_path = r"D:\khoaluan\data\data_luat_hon_nhan_gia_dinh.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Tạo Document list
raw_documents = []
for item in data:
    if not item.get("noi_dung"):
        continue
    metadata = {k: v for k, v in item.items() if k != "noi_dung"}
    raw_documents.append(Document(page_content=item["noi_dung"], metadata=metadata))



 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(raw_documents)
uuids = [f"id{i}" for i in range(1, len(documents) + 1)]


for i in tqdm(range(0, len(documents), 100)):
    batch_docs = documents[i:i+100]
    batch_ids = uuids[i:i+100]
    try:
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
    except Exception as e:
        print(f"Lỗi ở batch {i}-{i+100}: {e}")



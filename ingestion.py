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

# Initialize index with host fallback if needed
index_host = os.environ.get("PINECONE_INDEX_HOST")
try:
    if index_host:
        index = pc.Index(index_name, host=index_host)
    else:
        desc = pc.describe_index(index_name)
        host = desc.get("host") if isinstance(desc, dict) else getattr(desc, "host", None)
        index = pc.Index(index_name, host=host) if host else pc.Index(index_name)
except Exception:
    index = pc.Index(index_name)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# initialize embeddings model + vector store

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# loading the documents from local folder
from langchain_community.document_loaders import PyPDFDirectoryLoader

docs_dir = os.environ.get("DOCS_DIR", "/workspace/documents")
loader = PyPDFDirectoryLoader(docs_dir)
raw_documents = loader.load()



 
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



<h1>Retrieval Augmented Generation (RAG) with Streamlit, LangChain and Pinecone</h1>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>

<h2>Installation</h2>
1. Clone the repository:

```
git clone https://github.com/ThomasJanssen-tech/LangChain-Pinecone-RAG.git
cd LangChain-Pinecone-RAG
```

2. Create a virtual environment

```
python -m venv venv
```

3. Activate the virtual environment

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

4. Install libraries

```
pip install -r requirements.txt
```

5. Create accounts

- Create a free account on Pinecone: https://www.pinecone.io/
- Create an API key for OpenAI: https://platform.openai.com/api-keys

6. Add API keys to .env file

- Rename .env.example to .env
- Add the API keys for Pinecone and OpenAI to the .env file

<h3>Executing the scripts</h3>

1. Open a terminal in VS Code

2. Execute the following command:

```
python sample_ingestion.py
python sample_retrieval.py
python ingestion.py
python retrieval.py
streamlit run chatbot_rag.py
```

Điều kiện kết hôn là gì?
Nhà nước khuyến khích phụ nữ sinh con trong độ tuổi nào?
Làm sao để đăng kí nuôi con nuôi?

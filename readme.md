<h1>Retrieval Augmented Generation (RAG) with Streamlit, LangChain and Pinecone</h1>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>

<h2>Installation</h2>
1. Clone the repository:

```
git clone https://github.com/ThomasJanssen-tech/Ollama-Chatbot.git
cd LangChain Pinecone RAG
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

<h3>Executing the scripts</h3>

1. Open a terminal in VS Code

2. Execute the following command:

```
python ingestion.py
python retrieval.py
streamlit run chatbot_rag.py
```

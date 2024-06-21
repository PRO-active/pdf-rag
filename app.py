import streamlit as st
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

st.title('PDF Q&A System using LangChain and OpenAI')

pdf_url = st.text_input("Enter the URL of the PDF:", "")


@st.cache_data
def load_pdf(url):
    response = requests.get(url)
    with open("paper.pdf", "wb") as f:
        f.write(response.content)
    loader = PyPDFLoader("paper.pdf")
    documents = loader.load()
    return documents

with st.spinner("Loading PDF..."):
    documents = load_pdf(pdf_url)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

@st.cache_resource
def create_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

with st.spinner("Creating vector store..."):
    vector_store = create_vector_store(docs)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following documents to answer the question in japanese. {context}.")
])

llm = ChatOpenAI(model="gpt-3.5-turbo")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

question = st.text_input("Enter your question here:", "")

if question:
    with st.spinner("Processing your question..."):
        result = qa_chain({"query": question})
        st.write("### Answer")
        st.write(result["result"])

        st.write("### Source Documents")
        for doc in result["source_documents"]:
            st.write(doc.page_content[:500])
FROM python:3.9

WORKDIR /app

RUN pip -q install streamlit openai langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu tiktoken pypdf

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
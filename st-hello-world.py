import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

local_path = "food_recommendations.pdf"

# Local PDF file uploads
if local_path:
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
else:
    st.error("Upload a PDF file")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=480, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="localrag"
)

local_model = "mistral"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate three
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to Generate restaurant recommendations prioritizing specified cuisine. Limit: 30 words. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

st.title("Bay Area Food Chatbot :hamburger:")

with st.form(key='my_form'):
    text_input = st.text_input(label='Write a query asking about food in the Bay Area:')
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        output_text = chain.invoke({"context": "", "question": text_input})
        # Remove newline characters from the output text
        clean_output_text = output_text.replace('\n', '')
        # Print clean output text
        st.write(clean_output_text)



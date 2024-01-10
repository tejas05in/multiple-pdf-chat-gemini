import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.

    Args:
        pdf_docs (List[Path]): A list of paths to PDF documents.

    Returns:
        str: The concatenated text from the PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Splits a large text into smaller chunks.

    Args:
        text (str): The text to be split.

    Returns:
        List[str]: A list of smaller chunks of text.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store from a list of text chunks.

    Args:
        text_chunks (List[str]): A list of text chunks.

    Returns:
        FAISS: The FAISS vector store.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():
    """
    Returns a question-answering chain that can be used for conversational tasks.

    The chain is trained on the Stable Diffusion dataset and can be used to generate answers to questions based on a given context.

    The chain takes two inputs: "context" and "question". The "context" input should be a string containing the context in which the question is asked, while the "question" input should be a string containing the question itself.

    The chain outputs a string containing the answer to the question. If the answer is not available in the given context, the chain will output a message saying "Answer is not available in the provided context".

    Note that the chain may return a wrong answer sometimes, so it is important to ensure that the answer is correct before using it in a real-world application.

    Args:
        None

    Returns:
        A question-answering chain that can be used for conversational tasks.
    """
    prompt_template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    the provided context just say, "Answer is not available in the provided context" , do not provide the wrong answer \n\n
    
    Context:\n {context}\n
    Question:\n {question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    """
    This function takes a user input question and returns the response from the trained model.

    Args:
        user_question (str): The question asked by the user.

    Returns:
        str: The response from the trained model.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs using GEMINI - PRO")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Proces"):
            with st.spinner("Procesing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("DONE!")


if __name__ == "__main__":
    main()

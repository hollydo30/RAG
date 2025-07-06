from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import gradio as gr
import os

# Load lightweight FLAN-T5 Small model
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Global RAG components
vectorstore = None
qa_chain = None

# Step 1: Load and process PDF from path
def load_pdf_to_vectorstore(pdf_path):
    global vectorstore, qa_chain

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"[Errno 2] No such file: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Step 2: Ask questions
def ask_pdf_question(query):
    if qa_chain is None:
        return "⚠️ PDF not loaded. Please check the backend code."
    return qa_chain.run(query)

# Call PDF loader here (backend-controlled path)
# Replace this with the actual path to your PDF file
pdf_path = "C:/Users/holly/OneDrive/Desktop/Resume_I.pdf"
load_pdf_to_vectorstore(pdf_path)

# Gradio UI – just the Q&A part
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Ask Questions About the PDF")

    with gr.Row():
        question_input = gr.Textbox(label="Ask a question about the PDF")
        answer_output = gr.Textbox(label="Answer", lines=4)

    question_input.submit(fn=ask_pdf_question, inputs=question_input, outputs=answer_output)

demo.launch()


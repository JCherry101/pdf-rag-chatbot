import os
from dotenv import load_dotenv # Use dotenv for local API key management
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# HuggingFace and Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain v1.0.1+ core modules
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough # Added RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser # Added for clean output

# --- 1. Load Google API Key from .env file ---
load_dotenv() # Reads the .env file in your project folder
print("Attempting to load Google API Key from .env file...")
if not os.getenv("GOOGLE_API_KEY"):
    # Check if the key was loaded
    raise SystemExit("Error: GOOGLE_API_KEY not found in .env file. Please create the file and add your key.")
print("Google API Key Loaded.")

try:
    # Use the same Gemini model, ensure API key is available
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1) # Corrected model name from last time
    print("Gemini LLM Initialized.")

    # Use the same local Hugging Face embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Hugging Face Embeddings Initialized.")
except Exception as e:
    print(f"Error initializing models: {e}")
    # Provide more specific advice for common API key issues
    if "api_key" in str(e).lower():
        print("Double-check that your GOOGLE_API_KEY is correct in the .env file.")
    raise SystemExit("Model initialization failed.") from e

# --- 3. Load and Split PDF ---
pdf_file_path = "company_handbook.pdf" # Assumes PDF is in the same folder
if not os.path.exists(pdf_file_path):
    raise SystemExit(f"Error: PDF file not found at '{pdf_file_path}'. Please place it in the project folder.")

try:
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    if not docs:
            raise ValueError("No documents loaded from PDF. It might be empty or corrupted.")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    print(f"Loaded and split '{pdf_file_path}' into {len(split_docs)} chunks.")
except Exception as e:
    print(f"Error loading/splitting PDF: {e}")
    raise SystemExit("PDF processing failed.") from e


# --- 4. Create Vector Database ---
try:
    vector_store = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_store.as_retriever()
    print("Vector database created.")
except Exception as e:
    print(f"Error creating vector store: {e}")
    raise SystemExit("Vector store creation failed.") from e


# --- 5. Create RAG Chain (LCEL Runnable Architecture) ---

# Define how to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the user's question based ONLY on the following context:

    <context>
    {context}
    </context>

    Question: {question}

    Answer:
    """
)

# Build the RAG chain using LangChain Expression Language (LCEL)
rag_chain = (
    # RunnablePassthrough allows us to pass the original question through
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser() # Parses the LLM's message output into a simple string
)

print("--- RAG System Ready! ---")

# --- 6. Interactive Q&A Loop ---
if __name__ == "__main__":
    while True:
        try:
            my_question = input("\n Ask a question about the document (or type 'exit'): ")
            if my_question.lower() in ['exit', 'quit']:
                break
            if not my_question:
                continue

            print(" Thinking...")
            # Invoke the chain with the user's question
            answer = rag_chain.invoke(my_question)
            print("\n Answer:")
            print(answer)

        except Exception as e:
            print(f"An error occurred during query: {e}")

    print("\n Exiting RAG system.")
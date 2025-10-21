# Simple RAG Chatbot with PDF

This script allows you to chat with a PDF document using Google Gemini and local embeddings.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Add API Key:** Create a file named `.env` in the project root and add your Google AI API key:
    ```
    GOOGLE_API_KEY="AIza..."
    ```
5.  **Add PDF:** Place your PDF file in the project root and name it `company_handbook.pdf`.

## Running

Ensure your virtual environment is active, then run:
```bash
python ask_my_pdf_local.py
from flask import Flask, request, jsonify, send_from_directory
from ask_my_pdf import rag_chain # Imports your existing configured RAG chain

app = Flask(__name__)

# Serve the simple frontend HTML file
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# API endpoint to process questions
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Pass the user's question to your existing LangChain RAG pipeline
        answer = rag_chain.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting web server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

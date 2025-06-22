# Medical RAG QA Application

This is a Retrieval-Augmented Generation (RAG) application designed to answer medical questions. It uses a local language model and a vector database to provide answers based on a given set of documents, and can also process user-uploaded PDFs for on-the-fly querying.

## Project Structure

- **/Frontend**: Contains the `index.html` file for the user interface.
- **/Backend**: Contains the core application logic, including the FastAPI server, models, and data.

## Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Set Up the Backend

All backend commands should be run from within the `Backend` directory.

```bash
cd Backend
```

#### a. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate
```

#### b. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

#### c. Download the Language Model

Run the `download_model.py` script to download the GGUF model file. This is a large file (~4GB), so the download may take some time.

```bash
python download_model.py
```

### 3. Prepare the Data (Optional)

If you want to query a default set of documents, place your PDF files into the `Backend/Data` directory. The application will use these if no PDF is uploaded during a query. If this directory does not exist, you can create it.

### 4. Run the Application

Once the setup is complete, you can start the FastAPI server.

```bash
python main.py
```

The server will start, and you can access the application in your web browser at:

**[http://127.0.0.1:8001](http://127.0.0.1:8001)**

You can now ask questions, upload your own PDFs, and receive answers from the model.

## Features

- Medical document ingestion and processing
- Vector-based semantic search using Chroma
- Question answering powered by Meditron 7B LLM
- Web interface for easy interaction
- PubMedBERT embeddings for medical text

## Project Structure

- `rag.py`: Main application file with web interface
- `ingest.py`: Document ingestion and processing
- `retriever.py`: Vector search implementation
- `download_model.py`: Script to download Meditron model
- `templates/`: HTML templates for web interface
- `static/`: Static files (CSS, JS)
- `chroma_db/`: Vector database storage
- `models/`: Downloaded model files

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for Python dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

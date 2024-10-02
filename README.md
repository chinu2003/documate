# DocuMate - Smart PDF Query Platform 🦜📄
### Using Open Source Language Models

DocuMate is a smart PDF query platform designed to help users upload PDFs and extract valuable insights by asking natural language questions. It supports extracting text, tables, and images from PDFs, using open-source language models to generate accurate answers. This makes document interaction efficient, user-friendly, and accessible.

## Features
- **PDF Upload:** Upload your PDFs and extract information from text, tables, and images.
- **Table and Image Extraction:** Extracts structured data from tables and images within the PDF using `pdfplumber`.
- **Natural Language Queries:** Interact with your PDF by asking questions in natural language.
- **Voice Interaction:** Supports voice input for asking questions.
- **Text-to-Speech Output:** Converts text responses into audio using gTTS.

## Installation

### Step 1: Clone the Hugging Face LaMini-T5-738M Model
To use the model for question answering, first clone the LaMini-T5-738M repository from Hugging Face:

```bash
git clone https://huggingface.co/MBZUAI/LaMini-T5-738M

Step 2: Clone the Project Repository
Next, clone the DocuMate project repository to your local machine:
git clone <repository_url>

Step 3: Create Necessary Folders
Navigate into the project directory and create the following folders for storing embeddings, models, and documents:
mkdir db
mkdir models
mkdir docs

Step 4: Add Model Files
After cloning the Hugging Face LaMini-T5-738M model, move the model files into the models folder within the project.

Usage
Step 1: Run the Ingestion Script
Run the ingest.py script to process and prepare the PDF data for querying:
python ingest.py


Here’s the full README.md file based on the information you provided:

markdown
Copy code
# DocuMate - Smart PDF Query Platform 🦜📄
### Using Open Source Language Models

DocuMate is a smart PDF query platform designed to help users upload PDFs and extract valuable insights by asking natural language questions. It supports extracting text, tables, and images from PDFs, using open-source language models to generate accurate answers. This makes document interaction efficient, user-friendly, and accessible.

## Features
- **PDF Upload:** Upload your PDFs and extract information from text, tables, and images.
- **Table and Image Extraction:** Extracts structured data from tables and images within the PDF using `pdfplumber`.
- **Natural Language Queries:** Interact with your PDF by asking questions in natural language.
- **Voice Interaction:** Supports voice input for asking questions.
- **Text-to-Speech Output:** Converts text responses into audio using gTTS.

## Installation

### Step 1: Clone the Hugging Face LaMini-T5-738M Model
To use the model for question answering, first clone the LaMini-T5-738M repository from Hugging Face:

```bash
git clone https://huggingface.co/MBZUAI/LaMini-T5-738M
Step 2: Clone the Project Repository
Next, clone the DocuMate project repository to your local machine:

git clone <repository_url>
Step 3: Create Necessary Folders
Navigate into the project directory and create the following folders for storing embeddings, models, and documents:


mkdir db
mkdir models
mkdir docs
Step 4: Add Model Files
After cloning the Hugging Face LaMini-T5-738M model, move the model files into the models folder within the project.

Usage
Step 1: Run the Ingestion Script
Run the ingest.py script to process and prepare the PDF data for querying:
python ingest.py


Step 2: Start the Chatbot Application
Use Streamlit to launch the chatbot interface, where you can upload PDFs and ask questions:
python -m streamlit run chatapp.py


References and Acknowledgments
Special thanks to AI Anytime for inspiration and insights.

License
This project is open-source and is available under the MIT License.```
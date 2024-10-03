from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import os
import PyPDF2
import validators
from datetime import datetime
import faiss
import pickle
from embedding import text_embedding, query_embedding
from azure_blob_storage import get_data_from_blob, getEmbeddingFiles
from azure_files_share import getFiles, uploadFile_inazure, checkFileInAzure
import io
import numpy as np
from pdf2image import convert_from_path
import easyocr
import tempfile
import time

load_dotenv()
MAX_SIZE_MB = 2 * 1024 * 1024  # 2 MB in bytes
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = 'your_unique_secret_key123'  # Replace with a secure random string
CORS(app)

UPLOAD_FOLDER = 'uploaded_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
reader = easyocr.Reader(['en'])

model = 'models/embedding-001'
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

prompt_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction="Your name is Angel. Your role is to find the best and most relevant answer with step-by-step instructions to the user's question."
)

@app.route('/')
def index():
    return 'working'

###############################################################

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Try to get the user's message from the JSON payload
        user_message = request.json.get('message')
        file_name = request.json.get('fileName')
        indexfile = file_name + '.index'
        metadataFile = file_name + '.pkl'

        if not file_name:
            if not user_message:
                return jsonify({'error': 'Message is required.'}), 400

            # Create a prompt to send to the model
            prompt = f"Please find the best answer to my question.\nQUESTION - {user_message}"
            
            # Generate response from the model
            response = prompt_model.generate_content(prompt)
            if not response or not response.text:
                return jsonify({'error': 'Failed to generate a response.'}), 500

            return jsonify({'response': response.text})
        else:
            if not user_message:
                return jsonify({'error': 'Message is required.'}), 400

            # Embed the user message
            query_embed = query_embedding(user_message)
            
            # Retrieve the index from Azure Blob as bytes
            index_bytes = get_data_from_blob(indexfile, "vectorsfiles")
            if not index_bytes:
                return jsonify({'error': 'FAISS index not found.'}), 404

            # Deserialize the FAISS index from bytes
            try:
                index = faiss.deserialize_index(index_bytes)
            except Exception as e:
                return jsonify({'error': f'Failed to deserialize FAISS index: {str(e)}'}), 500

            # Load metadata (deserialize using pickle)
            try:
                metadata_bytes = get_data_from_blob(metadataFile, "metadafiles")
                if not metadata_bytes:
                    return jsonify({'error': 'Metadata file not found.'}), 404
                if isinstance(metadata_bytes, str):
                    metadata_bytes = metadata_bytes.encode('utf-8')
                metadata = pickle.loads(metadata_bytes)  # Deserialize metadata from bytes
            except Exception as e:
                return jsonify({'error': f'Failed to load metadata: {str(e)}'}), 500

            if metadata is None:
                return jsonify({'error': 'Metadata is empty or None.'}), 500

            # Perform search on the FAISS index
            k = 3  # Retrieve top 3 results
            try:
                # Ensure query_embed is a 2D array
                query_embed_np = np.array([query_embed]).astype('float32')
                distances, indices = index.search(query_embed_np, k)
            except Exception as e:
                return jsonify({'error': f'Failed to search FAISS index: {str(e)}'}), 500

            # Retrieve the metadata for the nearest neighbors
            relevant_chunks = ''
            for idx in indices[0]:
                if idx != -1:  # Check if valid index
                    relevant_chunks += metadata[idx]['text'] + "\n"

            # Create prompt for the model
            INSTRUCTION = "If the user asks for personal information such as patient name, license number, personal name, investigator officer name, or other sensitive information, respond with 'XYZ is the name or number, sorry, I do not provide personal information.'"
            makePrompt = f"PARAGRAPH - {relevant_chunks}\nUSER QUESTION - {user_message}\n{INSTRUCTION}"
            prompt = "Please find the best answer to the user's question from the given paragraph.\n" + makePrompt

            # Generate response from the model
            response = prompt_model.generate_content(prompt)

            if not response or not response.text:
                return jsonify({'error': 'Failed to generate a response.'}), 500

            return jsonify({'response': response.text})

    except KeyError as e:
        return jsonify({'error': f'Missing key in request: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Value error: {str(e)}'}), 400
    except Exception as e:
        # Catch any other unexpected errors
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

##############################################

@app.route('/files_get', methods=['GET'])
def getAllFiles():
    data = getFiles()
    return jsonify(data)

################################################

@app.route('/metadata_file', methods=['GET'])
def getMetadataFile():
    data = getEmbeddingFiles()
    return jsonify(data)

##############################################

def split_into_chunks(input_string, chunk_size=250):
    return [input_string[i:i + chunk_size] for i in range(0, len(input_string), chunk_size)]

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ''
            for page in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page].extract_text()
                if page_text:
                    text += page_text
        # Check if any text was extracted
        if text.strip():
            return text
        else:
            return 'ERROR'
    except Exception as e:
        print(f"An error occurred: {e}")
        return 'ERROR'

##############################################
##############################################
def extract_text_with_ocr(file_path):
    try:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        images = convert_from_path(file_path)
        pdftext = ''
        output = "ocr_img"
        imgpath = []
        os.makedirs(output, exist_ok=True)

        for i, image in enumerate(images):
            imagepath = os.path.join(output, f'page_{i+1}.png')
            image.save(imagepath, "PNG")
            imgpath.append(imagepath)

        for image_path in imgpath:
            results = reader.readtext(image_path)
            # Append extracted text
            for result in results:
                text = result[1]
                pdftext += text + " "

        return pdftext

    except Exception as e:
        print(f"Error processing PDF {file_path}: {str(e)}")
        return f"Error processing PDF {file_path}: {str(e)}"

##############################################
##############################################

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        ocr_choice = request.form.get('ocr')  # Assuming OCR choice is passed in form data
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        file_name = file.filename
        fname = os.path.splitext(file_name)[0]
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        uploaded_file_res = checkFileInAzure(file_name)  # Get existing files

        # Check if the file already exists
        if uploaded_file_res == 'EXISTS':
            return jsonify({'error': 'File already exists.'}), 202

        # Save the uploaded file to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        if ocr_choice == 'true':
            text = extract_text_with_ocr(temp_file_path)
            if text.startswith("Error processing PDF"):
                os.unlink(temp_file_path)
                return jsonify({'error': text}), 500
            chunks = split_into_chunks(text)
            embeddings = text_embedding(chunks, fname)  # Assuming this returns embeddings list

            # Create FAISS index
            dimension = len(embeddings[0]) if embeddings else 0
            if dimension == 0:
                os.unlink(temp_file_path)
                return jsonify({'error': 'No embeddings generated.'}), 500
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype('float32'))

            # Serialize FAISS index
            index_bytes = faiss.serialize_index(index)

            # Create metadata
            metadata = [{'text': chunk} for chunk in chunks]

            # Serialize metadata
            metadata_bytes = pickle.dumps(metadata)

            # Upload index and metadata to Azure Blob
            indexfile = fname + '.index'
            metadataFile = fname + '.pkl'
            upload_success_index = uploadFile_inazure(io.BytesIO(index_bytes), indexfile, "vectorsfiles")
            upload_success_metadata = uploadFile_inazure(io.BytesIO(metadata_bytes), metadataFile, "metadafiles")
            if not upload_success_index or not upload_success_metadata:
                os.unlink(temp_file_path)
                return jsonify({'error': 'Failed to upload index or metadata to Azure.'}), 500

            # Upload the original PDF to Azure
            file.seek(0)  # Reset file pointer
            upload_success_file = uploadFile_inazure(file, file_name, "pdffiles")  # Assuming 'pdffiles' is the container
            if not upload_success_file:
                os.unlink(temp_file_path)
                return jsonify({'error': 'Failed to upload file to Azure.'}), 500

            # Clean up temporary files and OCR images
            os.unlink(temp_file_path)
            output = "ocr_img"
            if os.path.exists(output):
                for img_file in os.listdir(output):
                    if img_file.endswith('.png'):
                        os.remove(os.path.join(output, img_file))

            return jsonify({'message': f'File {file_name} uploaded with OCR processing'}), 200
        else:
            # Process file without OCR
            text = extract_text_from_pdf(temp_file_path)
            if text == 'ERROR':
                os.unlink(temp_file_path)
                return jsonify({'error': 'Failed to extract text from PDF.'}), 203

            chunks = split_into_chunks(text)
            embeddings = text_embedding(chunks, fname)  # Assuming this returns embeddings list

            # Create FAISS index
            dimension = len(embeddings[0]) if embeddings else 0
            if dimension == 0:
                os.unlink(temp_file_path)
                return jsonify({'error': 'No embeddings generated.'}), 500
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype('float32'))

            # Serialize FAISS index
            index_bytes = faiss.serialize_index(index)

            # Create metadata
            metadata = [{'text': chunk} for chunk in chunks]

            # Serialize metadata
            metadata_bytes = pickle.dumps(metadata)

            # Upload index and metadata to Azure Blob
            indexfile = fname + '.index'
            metadataFile = fname + '.pkl'
            upload_success_index = uploadFile_inazure(io.BytesIO(index_bytes), indexfile, "vectorsfiles")
            upload_success_metadata = uploadFile_inazure(io.BytesIO(metadata_bytes), metadataFile, "metadafiles")
            if not upload_success_index or not upload_success_metadata:
                os.unlink(temp_file_path)
                return jsonify({'error': 'Failed to upload index or metadata to Azure.'}), 500

            # Upload the original PDF to Azure
            file.seek(0)  # Reset file pointer
            upload_success_file = uploadFile_inazure(file, file_name, "pdffiles")  # Assuming 'pdffiles' is the container
            if not upload_success_file:
                os.unlink(temp_file_path)
                return jsonify({'error': 'Failed to upload file to Azure.'}), 500

            # Clean up temporary file
            os.unlink(temp_file_path)

            return jsonify({'message': f'File {file_name} uploaded without OCR processing'}), 200

    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

##############################################################

@app.route('/load_file', methods=['POST'])
def select_file():
    # If /load_file is not used, it can be removed or implemented properly
    # For now, return a message indicating it's not implemented
    return jsonify({'message': 'load_file endpoint is not implemented yet.'}), 501

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000))

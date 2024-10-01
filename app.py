from flask import Flask,session, request, Response,jsonify
from flask_cors import CORS
import time
from dotenv import load_dotenv
import google.generativeai as genai
import json
import os
import PyPDF2
import validators
from datetime import datetime
import faiss
import pickle
from embedding import text_embedding,query_embedding
from azure_blob_storage import get_data_from_blob,getEmbeddingFiles
from azure_files_share import getFiles,uploadFile_inazure,checkFileInAzure
import io
import numpy as np
from pdf2image import convert_from_path
import easyocr

load_dotenv()
MAX_SIZE_MB = 2 * 1024 * 1024  # 5 MB in bytes
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = 'your_unique_secret_key123'  # Replace with a secure random string
CORS(app)

UPLOAD_FOLDER = 'uploaded_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
reader = easyocr.Reader(['en'])
# Load the model
# model = SentenceTransformer('all-MiniLM-L6-v2')
# a = ['To \nRegional Passport Office  \nBhopal, Behind Paryavas Bhawan  \nArera Hills, Bhopal - 4562011 (M.P.)  \n \nSubject: Response to Passport Application Nos. BP7068162817423 and BP7068162763623  \n \nDear Sir,  \n \nI am Irfan Khan, and my mother is Haseena Kh', 'an. We both applied for passports at the same time. \nAt the first stage, our document verification was successfully processed via the passport application \nfrom Dewas. However, during the second stage, which is add ress verification, our applications', ' were \nnot successfully processed by the Civil Line Police Station, Dewas.  \n \nThe reason for this is that I am currently staying in a rented accommodation, and my Aadhaar card is \nregistered at my old address: 43 EWS Nagar Ni gam Colony. The landlor', 'd does not provide any rent \nagreement, which led to the failure of our address verification.  \n \nCurrently, I am residing at 155 Vikram Nagar, Dewas, and I have updated my address in my Aadhaar \ncard. However, I could not update the addr ess in my v', 'oter ID due to the election time constraints. I \nhave attached the necessary documents for both myself and my mother with this letter.  \n \nI kindly request that you process our applications and, for the purpose of address verification, \narrange for t', ' he police to verify our address at 155 Vikram Nagar, Dewas. Your assistance in this \nmatter would be greatly appreciated.  \n \nThank you.  \nSincerely,  \nIrfan Khan  \nMobille no. – 8889803337  \n \nIrfan Khan s ignature  - \nHaseena Khan signature - ']
# chunk_embeddings = model.encode(a)
# print(chunk_embeddings)



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
  safety_settings = safety_settings,
  # See https://ai.google.dev/gemini-api/docs/safety-settings
  system_instruction="Your name is Angel. Your role is to find the best and most relevant answer with step by step to the user's question.",
)


@app.route('/')
def index():
    return 'working'
###############################################################

@app.route('/chat', methods=['POST'])
def chat():
    # Try to get the user's message from the JSON payload
    user_message = request.json.get('message')
    file_name = request.json.get('fileName')
    indexfile = file_name + '.index'
    metadataFile = file_name + '.pkl'

    if not file_name:
        if not user_message:
            return jsonify({'error': 'Message is required.'}), 400

        # Create the prompt for the model
        prompt = "Please find the best answer to my question.\nQUESTION - " + user_message
        
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
        # Deserialize the FAISS index from bytes
        #try:
        index_array = np.frombuffer(index_bytes, dtype=np.uint8)
        index = faiss.deserialize_index(index_array)
        #except Exception as e:
            #return jsonify({'error': f'Failed to load FAISS index: {str(e)}'}), 500

        # Load metadata (deserialize using pickle)
        time.sleep(2)

        metadata_bytes = get_data_from_blob(metadataFile, "metadafiles")
        #try:
        if isinstance(metadata_bytes, str):
            metadata_bytes = metadata_bytes.encode('utf-8')
        metadata = pickle.loads(metadata_bytes)  # Deserialize metadata from bytes
        #except Exception as e:
        #    return jsonify({'error': f'Failed to load metadata: {str(e)}'}), 500

        if metadata is None:
            return jsonify({'error': 'Metadata is empty or None.'}), 500

        # Perform search on the FAISS index
        k = 3  # Retrieve top 3 results
        try:
            distances, indices = index.search(np.array(query_embed).astype('float32'), k)
        except Exception as e:
            return jsonify({'error': f'Failed to search FAISS index: {str(e)}'}), 500

        # Retrieve the metadata for the nearest neighbors
        relevant_chunks = ''
        for idx in indices[0]:
            if idx != -1:  # Check if valid index
                relevant_chunks += metadata[idx]['text']

        # Create prompt for the model
        INSTRUCTION = "If the user asks for personal information such as patient name, license number, personal name, investigator officer name, or other sensitive information, respond with 'XYZ is the name or number, sorry, I do not provide personal information.'"
        makePrompt = f"PARAGRAPH - {relevant_chunks}\nUSER QUESTION - {user_message}\n{INSTRUCTION}"
        prompt = "Please find the best answer to the user's question from the given paragraph.\n" + makePrompt

        # Generate response from the model
        response = prompt_model.generate_content(prompt)

        if not response or not response.text:
            return jsonify({'error': 'Failed to generate a response.'}), 500

        return jsonify({'response': response.text})


    # except KeyError as e:
    #     return jsonify({'error': f'Missing key in request: {str(e)}'}), 400
    # except ValueError as e:
    #     return jsonify({'error': f'Value error: {str(e)}'}), 400
    # except Exception as e:
    #     # Catch any other unexpected errors
    #     return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

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

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()

        # Check if any text was extracted
        if text.strip():
            return text
        else:
            return 'ERROR'     
    except Exception as e:
        print(f"An error occurred: {e}")
##############################################
##############################################
def extract_text_with_ocr(file_name):
    try:
        filename = os.path.splitext(file_name)[0]
        images = convert_from_path(file_path)
        pdftext=''
        output = "ocr_img"
        imgpath = []
        os.makedirs(output,exist_ok=True)

        for i , image in enumerate(images):
            imagepath = os.path.join(output,f'page_{i+1}.png')
            image.save(imagepath,"PNG")
            imgpath.append(imagepath)

        for image_path in imgpath:
            results = reader.readtext(image_path)
        # Print extracted text
            for result in results:
                text = result[1]
                pdftext +=text

        return pdftext

    except Exception as e:
        r = f"Error processing PDF {file_name}: {str(e)}"
        return r    
    

##############################################
##############################################

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    ocr_choice = request.form.get('ocr')  # Assuming OCR choice is passed in form data
    file_name = file.filename
    fname = os.path.splitext(file_name)[0]
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Check if the file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 201

    uploaded_file_res = checkFileInAzure(file_name)  # Get existing files

    # Check if the file already exists
    if uploaded_file_res == 'EXISTS':
        return jsonify({'error': 'File already exists.'}), 202

    file_in_memory = io.BytesIO(file.read())

    if ocr_choice == 'true':
        text = extract_text_with_ocr(file_in_memory)
        chunks = split_into_chunks(text)
        text_embedding(chunks,fname)
        res = uploadFile_inazure(file)

        for file in os.listdir('ocr_img'):
            if file.endswith('.png'):
                os.remove(os.path.join(directory, file))
        # Perform OCR processing on the file (Add your OCR processing logic here)
        return jsonify({'message': f'File {file_name} uploaded with OCR processing'}), 200
    else:
        # Process file without OCR
        text = extract_text_from_pdf(file_in_memory)
        if text == 'ERROR':
            return jsonify({'error': 'File already exists.'}), 203          
        #file_in_memory.close()
        chunks = split_into_chunks(text)

        text_embedding(chunks,fname)
        res = uploadFile_inazure(file) ## Upload file in azure file share

        return jsonify({'message': f'File {file_name} uploaded without OCR processing'}), 200

##############################################################
@app.route('/load_file', methods=['POST'])
def select_file():
    data = request.json
    file_name = data.get('fileName')

    filename = file_name+'.index'
    #res = get_data_from_blob(filename,'vectorsfiles')
    #print(res)
    # index = load_faiss_index_metadata(filename,"vectorfiles")
    # print(index)

    # if res != 'error':
    #     raw_size_in_bytes = len(res)# 2 MB in bytes
    #     max_size_in_bytes = 2 * 1024 * 1024  # 2,097,152 bytes
    #     if raw_size_in_bytes < max_size_in_bytes:
    #         session['faiss_index'] = res

    return jsonify({'message': file_name})

    # #metadata = load_faiss_metadata(file_name,"metadatafiles")
    # if index is None:
    #     print('failed')
    #     return jsonify({'error': 'Failed to load FAISS index or metadata.'}), 500

    # # Check the file sizes
    # index_size = os.path.getsize(index)
    # #metadata_size = os.path.getsize(metadata)
    
    # total_size = index_size

    # # If the total size is less than 2 MB, store in session
    # if total_size < MAX_SIZE_MB:
    #     session['faiss_index'] = index
    #     message = f"FAISS index and metadata stored in session. Total size: {total_size} bytes."
    # else:
    #     message = f"Total size exceeds 2 MB (Size: {total_size} bytes). Data not stored in session."

    #return jsonify({'message': file_name})



if __name__ == '__main__':
    app.run(debug=True)




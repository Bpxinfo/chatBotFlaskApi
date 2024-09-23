from flask import Flask, request, Response,jsonify
from flask_cors import CORS
import time
from dotenv import load_dotenv
import google.generativeai as genai
import json
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

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

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message')
#     responses = "Hi i am fine"
        
#     return jsonify({'response':responses})
    
    #return Response(generate_response(), content_type='text/plain')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Attempt to get the user's message from the JSON payload
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'Message is required.'}), 400

        # Create the prompt for the model
        prompt = "Please find the best answer to my question.\nQUESTION - " + user_message
        
        # Attempt to generate a response from the model
        response = prompt_model.generate_content(prompt)
        
        if not response or not response.text:
            return jsonify({'error': 'Failed to generate a response.'}), 500

        # Return the generated response
        return jsonify({'response': response.text})
    
    except KeyError as e:
        return jsonify({'error': f'Missing key in request: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Value error: {str(e)}'}), 400
    except Exception as e:
        # Catch any other unexpected errors
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)




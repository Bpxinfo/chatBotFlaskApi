from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API with the API key from .env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Define the model and its configuration
prompt_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
    safety_settings=[
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ],
    system_instruction="Your name is Angel. Your role is to find the best and most relevant answer with step-by-step instructions to the user's question."
)

# Test route to ensure API is working
@app.route('/')
def index():
    return 'API is working'

# Chat route that processes user input and generates a response using the model
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the user's message from the request
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'Message is required.'}), 400

        # Create a prompt to send to the model
        prompt = f"Please find the best answer to my question.\nQUESTION - {user_message}"
        
        # Generate a response from the model
        response = prompt_model.generate_content(prompt)
        if not response or not response.text:
            return jsonify({'error': 'Failed to generate a response.'}), 500

        # Return the response in JSON format
        return jsonify({'response': response.text})

    except KeyError as e:
        return jsonify({'error': f'Missing key in request: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Value error: {str(e)}'}), 400
    except Exception as e:
        # Catch any other unexpected errors
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)




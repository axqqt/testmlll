from flask import Flask, request, jsonify, render_template
from sinhala_chatbot import SinhalaChatbot
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the chatbot instance with configurable path
knowledge_base_path = os.environ.get(
    'KNOWLEDGE_BASE_PATH', 'company_data.json')
chatbot = SinhalaChatbot(knowledge_base_path=knowledge_base_path)


@app.route('/')
def home():
    """Render the home page with a simple chat interface."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    API endpoint for interacting with the SinhalaChatbot.
    Expects a JSON payload with the key 'user_input'.
    Returns the chatbot's response in JSON format.
    """
    try:
        # Parse JSON data from the request
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({"error": "Missing 'user_input' in request body"}), 400

        user_input = data['user_input']
        logger.info(f"Received user input: {user_input}")

        # Process the user input using the chatbot
        response = chatbot.chat(user_input)

        # Return the chatbot's response
        return jsonify({
            "response": response,
            "language": "si" if chatbot.is_sinhala_text(user_input) else "en"
        }), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """
    API endpoint to clear the chatbot's conversation history.
    """
    try:
        chatbot.clear_history()
        return jsonify({"message": "Chat history cleared successfully"}), 200
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/api/add_knowledge', methods=['POST'])
def add_knowledge():
    """
    API endpoint to add new entries to the knowledge base.
    Expects a JSON payload with 'question' and 'answer' keys.
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'answer' not in data:
            return jsonify({"error": "Missing required fields in request body"}), 400

        question = data['question']
        answer = data['answer']

        success = chatbot.update_knowledge_base(question, answer)
        if success:
            return jsonify({"message": "Knowledge base updated successfully"}), 200
        else:
            return jsonify({"error": "Failed to update knowledge base"}), 500

    except Exception as e:
        logger.error(f"Error updating knowledge base: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({"status": "OK", "message": "Sinhala Chatbot API is running"}), 200

# Error handlers


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Create a simple HTML interface if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sinhala Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .chat-container {
            height: 400px;
            border: 1px solid #ccc;
            padding: 10px;
            margin-bottom: 10px;
            overflow-y: auto;
        }
        .input-container {
            display: flex;
        }
        #user-input {
            flex-grow: 1;
            padding: 8px;
            margin-right: 10px;
        }
        .message {
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e6f7ff;
            text-align: right;
        }
        .bot-message {
            background-color: #f0f0f0;
        }
        .button {
            padding: 8px 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h1>සිංහල චැට්බොට් - Sinhala Chatbot</h1>
    <p>Type in Sinhala or English. The chatbot will detect the language and respond accordingly.</p>
    
    <div class="chat-container" id="chat-container"></div>
    
    <div class="input-container">
        <input type="text" id="user-input" placeholder="Type your message here...">
        <button class="button" onclick="sendMessage()">Send</button>
        <button class="button" style="margin-left: 10px; background-color: #f44336;" onclick="clearHistory()">Clear History</button>
    </div>
    
    <script>
        function addMessage(text, isUser) {
            const chatContainer = document.getElementById('chat-container');
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message');
            messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
            messageDiv.textContent = text;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function sendMessage() {
            const inputField = document.getElementById('user-input');
            const userMessage = inputField.value.trim();
            
            if (userMessage) {
                addMessage(userMessage, true);
                inputField.value = '';
                
                fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ user_input: userMessage }),
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        addMessage('Error: ' + data.error, false);
                    } else {
                        addMessage(data.response, false);
                    }
                })
                .catch(error => {
                    addMessage('Error connecting to server. Please try again.', false);
                    console.error('Error:', error);
                });
            }
        }
        
        function clearHistory() {
            fetch('/api/clear_history', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                const chatContainer = document.getElementById('chat-container');
                chatContainer.innerHTML = '';
                addMessage('Chat history has been cleared.', false);
            })
            .catch(error => {
                addMessage('Error clearing chat history.', false);
                console.error('Error:', error);
            });
        }
        
        // Allow pressing Enter to send messages
        document.getElementById('user-input').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
            ''')

    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')
    app.run(host='0.0.0.0', port=port, debug=debug)

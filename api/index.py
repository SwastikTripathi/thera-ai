import os
from flask import Flask, request, jsonify, send_from_directory
from huggingface_hub import InferenceClient

app = Flask(__name__)

HF_API_KEY = os.environ.get("HF_API_KEY")
client = InferenceClient(token=HF_API_KEY)

conversation = [
    {
        "id": "system",
        "role": "system",
        "content": (
            "You are an AI therapist. Give **empathetic, supportive responses** focused only on mental health and well-being. "
            "Use *simple English words* and a warm, human-like tone—like a kind friend who’s also a professional. "
            "Keep responses short (under 100 tokens) and clear. "
            "Structure them with **bold** for emphasis, *italics* for a gentle tone, and - bullet points only when listing multiple items or when it truly helps understanding. "
            "Each response should: **validate the user’s feelings**, give *one easy tip*, and ask *one simple open-ended question*. "
            "Vary your empathetic openings, such as 'That sounds tough,' 'I can see how that’s hard,' or 'I’m here for you.' "
            "If the user asks something not about mental health, say: *'I’m here to help with how you feel. What’s on your mind today?'* "
            "Always sound non-judgmental and caring."
        )
    },
    {
        "id": "initial1",
        "role": "assistant",
        "content": "Hey there, I’m your AI therapist. How can I help you today?"
    },
    {
        "id": "initial2",
        "role": "assistant",
        "content": "*Note: I’m an AI, not a real therapist, but I’m here to listen and support you!*"
    }
]

@app.route('/')
def serve_index():
    return send_from_directory('../static', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        user_id = data.get('id', '')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        if not user_id:
            return jsonify({"error": "No message ID provided"}), 400
        
        conversation.append({"id": user_id, "role": "user", "content": user_message})
        
        if len(conversation) > 11:
            conversation.pop(1)
        
        prompt = f"<s>[INST] {conversation[0]['content']} The conversation so far is:\n"
        for turn in conversation[1:]:
            prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
        prompt += "Now, respond as the assistant with an empathetic, supportive message focused on mental health. [/INST]"
        
        response = client.text_generation(
            prompt=prompt,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_new_tokens=100,
            temperature=0.75,  # Changed from 0.7 to 0.75
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            stop=["\n\n"]
        )
        
        assistant_message = response.strip()
        assistant_id = str(len(conversation))
        conversation.append({"id": assistant_id, "role": "assistant", "content": assistant_message})
        
        return jsonify({"response": assistant_message, "id": assistant_id})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500

@app.route('/api/update_message', methods=['POST'])
def update_message():
    try:
        data = request.json
        message_id = data.get('id')
        new_content = data.get('content')
        if not message_id or not new_content:
            return jsonify({"error": "Missing id or content"}), 400
        
        for msg in conversation:
            if msg['id'] == message_id:
                msg['content'] = new_content
                break
        else:
            return jsonify({"error": "Message not found"}), 404
        
        return jsonify({"status": "updated"})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500

@app.route('/api/regenerate_after', methods=['POST'])
def regenerate_after():
    try:
        data = request.json
        message_id = data.get('id')
        if not message_id:
            return jsonify({"error": "No message ID provided"}), 400
        
        # Find the index of the message to truncate after
        for i, msg in enumerate(conversation):
            if msg['id'] == message_id:
                # Truncate the conversation after this message
                conversation[:] = conversation[:i+1]
                break
        else:
            return jsonify({"error": "Message not found"}), 404
        
        # Generate a new AI response based on the truncated conversation
        prompt = f"<s>[INST] {conversation[0]['content']} The conversation so far is:\n"
        for turn in conversation[1:]:
            prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
        prompt += "Now, respond as the assistant with an empathetic, supportive message focused on mental health. [/INST]"
        
        response = client.text_generation(
            prompt=prompt,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_new_tokens=100,  # Adjusted to match /chat for consistency
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,  # Updated to match /chat
            stop=["\n\n"]           # Added to match /chat
        )
        
        assistant_message = response.strip()
        assistant_id = str(len(conversation))
        conversation.append({"id": assistant_id, "role": "assistant", "content": assistant_message})
        
        return jsonify({"response": assistant_message, "id": assistant_id})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500

@app.route('/api/clear', methods=['POST'])
def clear():
    global conversation
    conversation = [
        {
            "id": "system",
            "role": "system",
            "content": (
                "You are an AI therapist. Give **empathetic, supportive responses** focused only on mental health and well-being. "
                "Use *simple English words* and a warm, human-like tone—like a kind friend who’s also a professional. "
                "Keep responses short (under 100 tokens) and clear. "
                "Structure them with **bold** for emphasis, *italics* for a gentle tone, and - bullet points only when listing multiple items or when it truly helps understanding. "
                "Each response should: **validate the user’s feelings**, give *one easy tip*, and ask *one simple open-ended question*. "
                "Vary your empathetic openings, such as 'That sounds tough,' 'I can see how that’s hard,' or 'I’m here for you.' "
                "If the user asks something not about mental health, say: *'I’m here to help with how you feel. What’s on your mind today?'* "
                "Always sound non-judgmental and caring."
            )
        },
        {
            "id": "initial1",
            "role": "assistant",
            "content": "Hey there, I’m your AI therapist. How can I help you today?"
        },
        {
            "id": "initial2",
            "role": "assistant",
            "content": "*Note: I’m an AI, not a real therapist, but I’m here to listen and support you!*"
        }
    ]
    return jsonify({"response": "Conversation cleared"})

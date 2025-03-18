from flask import Flask, request, jsonify, send_from_directory
import requests
import json
import os

app = Flask(__name__)

# ------------------------------
# AI Client Configuration
# ------------------------------
ARLIAI_API_KEY = os.environ.get("ARLIAI_API_KEY")
if not ARLIAI_API_KEY:
    raise ValueError("ARLIAI_API_KEY environment variable is not set")

# ------------------------------
# Data Models
# ------------------------------
class Session:
    def __init__(self, user_id):
        self.user_id = user_id
        self.messages = []          # List of conversation messages
        self.diagnosis = None       # Diagnosis details once formed
        self.therapy_plan = TherapyPlan()  # Coping strategies/recommendations
        self.completed = False      # Session conclusion flag

class UserProfile:
    def __init__(self, user_id, name, age, gender):
        self.user_id = user_id
        self.name = name
        self.age = age
        self.gender = gender
        self.sessions = []          # History of sessions

class Diagnosis:
    def __init__(self, emotions=None, conditions=None, severity=0, confidence=0):
        self.emotions = emotions or []
        self.conditions = conditions or []
        self.severity = severity
        self.confidence = confidence

class TherapyPlan:
    def __init__(self):
        self.strategies = []  # List of coping strategies

# ------------------------------
# Session Storage
# ------------------------------
sessions = {}

# ------------------------------
# Helper Functions
# ------------------------------
def get_or_create_session(user_id):
    """Retrieve an existing session or create a new one for the user."""
    if user_id not in sessions:
        sessions[user_id] = Session(user_id)
    return sessions[user_id]

def generate_ai_response(prompt, max_tokens=300, temperature=0.7):
    """
    Generate an AI response using the ARLIAI API.
    Note: Parameters are set on the ARLIAI dashboard; only prompt and model are passed here.
    """
    try:
        url = "https://api.arliai.com/v1/chat/completions"
        payload = json.dumps({
            "model": "Mistral-Nemo-12B-Instruct-2407",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AI therapist. Provide detailed, empathetic, and structured responses with clear headings: "
                        "'Your Feelings', 'Next Steps', and 'Reflection'. Include supportive language, actionable suggestions, and reflective questions. "
                        "Elaborate your answers to offer comprehensive guidance."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "stream": False
        })
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {ARLIAI_API_KEY}"
        }
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        print(f"API Request Error: {str(e)}")
        return "I'm having trouble connecting right now. Please try again."
    except Exception as e:
        print(f"AI Error: {str(e)}")
        return "I'm having trouble processing that right now. Please try again."

def check_for_crisis(message):
    """Check for urgent, high-risk language; return True if detected."""
    risk_keywords = ['suicide', 'kill myself', 'self-harm', 'end it all']
    for keyword in risk_keywords:
        if keyword in message.lower():
            return True
    return False

def classify_emotions(message):
    """Classify the user's emotions from a message using an AI call."""
    prompt = (
        f"<s>[INST] Analyze the following message and list the primary emotions in **Markdown** bullet points "
        f"(e.g., **anxiety**, **stress**, **sadness**, **grief**):\n\n"
        f"\"{message}\"\n\n"
        "Respond only with the list. [/INST]"
    )
    response = generate_ai_response(prompt)
    emotions = [line.strip("- ").strip() for line in response.splitlines() if line.startswith("-")]
    if not emotions:
        emotions = [response.strip()]
    return emotions

def diagnose(session):
    """Formulate a diagnosis based on the conversation history."""
    context = "\n".join([f"**{msg['role'].capitalize()}**: {msg['content']}" for msg in session.messages])
    prompt = (
        f"<s>[INST] Based on the conversation below, provide a **diagnosis summary** in **Markdown**. "
        "Include a list of potential **conditions**, a **severity** score (1-5), and a **confidence** percentage (0-100%).\n\n"
        f"### Conversation Log\n{context}\n\n"
        "**Diagnosis Summary:**\n- **Conditions:** \n- **Severity:** \n- **Confidence:** \n\n"
        "Fill in the details accordingly. [/INST]"
    )
    response = generate_ai_response(prompt)
    try:
        conditions_line = [line for line in response.splitlines() if line.startswith("- **Conditions:**")]
        severity_line = [line for line in response.splitlines() if line.startswith("- **Severity:**")]
        confidence_line = [line for line in response.splitlines() if line.startswith("- **Confidence:**")]

        conditions = conditions_line[0].split("**Conditions:**")[-1].strip() if conditions_line else "General stress"
        severity = int(severity_line[0].split("**Severity:**")[-1].strip()) if severity_line else 3
        confidence = int(confidence_line[0].split("**Confidence:**")[-1].strip().replace("%", "")) if confidence_line else 80
    except Exception as e:
        print(f"Diagnosis parsing error: {str(e)}")
        conditions = "General stress"
        severity = 3
        confidence = 80

    emotions = classify_emotions(context)
    session.diagnosis = Diagnosis(
        emotions=emotions,
        conditions=[cond.strip() for cond in conditions.split(",") if cond] if conditions else ["General stress"],
        severity=severity,
        confidence=confidence
    )

def generate_coping_strategies(diagnosis):
    """Generate tailored coping strategies based on the diagnosis details."""
    prompt = (
        f"<s>[INST] Based on the following diagnosis details in **Markdown**:\n\n"
        f"- **Emotions:** {', '.join(diagnosis.emotions)}\n"
        f"- **Conditions:** {', '.join(diagnosis.conditions)}\n"
        f"- **Severity:** {diagnosis.severity}\n\n"
        "Provide **three actionable coping strategies** in **Markdown**. Format your answer using bullet points and "
        "integrate a brief explanation for each suggestion. Please do not use explicit labels for the sections; instead, "
        "structure the response with user-friendly headings. [/INST]"
    )
    response = generate_ai_response(prompt)
    strategies = [line.strip("- ").strip() for line in response.splitlines() if line.startswith("-")]
    return strategies if strategies else [response.strip()]

# ------------------------------
# Flask Routes
# ------------------------------
@app.route('/')
def serve_index():
    return send_from_directory('../static', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        user_id = data.get('id', '')
        
        if not user_message or not user_id:
            return jsonify({"error": "Missing message or user ID"}), 400

        # Crisis check: if high-risk language is detected, provide an immediate safety response.
        if check_for_crisis(user_message):
            crisis_response = (
                "**Emergency Alert:** It sounds like you're in intense distress. "
                "If you feel unsafe or are in immediate danger, please call your local emergency services or crisis hotline immediately.\n\n"
                "Remember, you deserve help and support."
            )
            return jsonify({"response": crisis_response, "id": user_id})
        
        # Retrieve or create the user's session
        session = get_or_create_session(user_id)
        session.messages.append({"role": "user", "content": user_message})
        
        # After several exchanges, form a diagnosis and generate coping strategies if not already done.
        if len(session.messages) >= 5 and not session.diagnosis:
            diagnose(session)
            strategies = generate_coping_strategies(session.diagnosis)
            session.therapy_plan.strategies = strategies
        
        # Build conversation context in Markdown
        context = "\n".join([f"**{msg['role'].capitalize()}**: {msg['content']}" for msg in session.messages])
        
        # Prepare the AI prompt with user-focused headings
        if session.diagnosis:
            prompt = (
                f"<s>[INST] The conversation so far in **Markdown**:\n\n"
                f"{context}\n\n"
                "Given the **diagnosis** with conditions: "
                f"{', '.join(session.diagnosis.conditions)} and severity: **{session.diagnosis.severity}**, "
                "please respond in **Markdown** with an empathetic message that includes the following sections:\n\n"
                "- **Your Feelings:** Briefly acknowledge and validate your current emotional state.\n"
                "- **Next Steps:** Provide one or two coping suggestions or strategies naturally integrated into the text.\n"
                "- **Reflection:** Pose a thoughtful, open-ended question to guide further discussion.\n\n"
                "Use headers, bullet lists, **bold**, and *italics* as appropriate. [/INST]"
            )
        else:
            prompt = (
                f"<s>[INST] The conversation so far in **Markdown**:\n\n"
                f"{context}\n\n"
                "Please respond in **Markdown** with an empathetic and supportive message that includes the following sections:\n\n"
                "- **Your Feelings:** Validate your current emotional state.\n"
                "- **Next Steps:** Provide one or two useful suggestions seamlessly integrated into the text.\n"
                "- **Reflection:** Ask a reflective question to encourage further sharing.\n\n"
                "Avoid using explicit labels like 'Actionable Tip' or 'Open-Ended Question.' "
                "Use headers, bullet points, **bold** text, and *italics* where it helps clarity. [/INST]"
            )
        
        assistant_message = generate_ai_response(prompt)
        session.messages.append({"role": "assistant", "content": assistant_message})
        
        # Conclude the session if enough exchanges have taken place.
        if len(session.messages) >= 10 and not session.completed:
            session.completed = True
            assistant_message += (
                "\n\n### Session Conclusion\n"
                "*It seems we have covered a lot today. Would you like to explore further strategies or schedule another session?*"
            )
        
        return jsonify({"response": assistant_message, "id": user_id})
    
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500

@app.route('/api/clear', methods=['POST'])
def clear():
    try:
        global sessions
        sessions = {}
        return jsonify({"response": "All sessions have been cleared."})
    except Exception as e:
        print(f"Error in clear: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500

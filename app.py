from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import uuid
import re
import random
import socket


# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise Exception("Please set your GEMINI_API_KEY in a .env file or environment variable")

# Initialize Gemini client
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins — change for production security

# Crisis detection regex patterns
CRISIS_PATTERNS = [
    r"\bsuicid(e|al)\b",
    r"\bkill myself\b",
    r"\bwant to die\b",
    r"\bself[- ]?harm\b",
    r"\bend my life\b",
]

def detect_crisis(text: str) -> bool:
    text = text.lower()
    return any(re.search(pattern, text) for pattern in CRISIS_PATTERNS)

# Greeting detection
def is_greeting(text: str) -> bool:
    greetings = [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    ]
    text = text.lower()
    for greet in greetings:
        if re.search(rf"\b{re.escape(greet)}\b", text):
            return True
    return False

# Motivational triggers & messages
MOTIVATIONAL_TRIGGERS = [
    "motivated",
    "need motivation",
    "encourage me",
    "feeling down",
    "discouraged",
    "help me stay positive",
    "i'm struggling",
    "can't keep going",
    "lost hope"
]

MOTIVATIONAL_MESSAGES = [
    "Believe in yourself! Every day is a new opportunity to grow.",
    "You are stronger than you think. Keep going!",
    "Small steps lead to big changes. You’ve got this!",
    "Remember, tough times don’t last, but tough people do.",
    "Every challenge is a chance to become a better version of yourself.",
    "Your feelings are valid. Take it one moment at a time.",
    "You have the power to overcome things even when it feels hard.",
]

# PHQ-9 questions
PHQ9_QUESTIONS = [
    "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
    "Over the last 2 weeks, how often have you felt down, depressed, or hopeless?",
    "Over the last 2 weeks, how often have you had trouble falling or staying asleep, or sleeping too much?",
    "Over the last 2 weeks, how often have you felt tired or had little energy?",
    "Over the last 2 weeks, how often have you had poor appetite or overeating?",
    "Over the last 2 weeks, how often have you felt bad about yourself — or that you are a failure or have let yourself or your family down?",
    "Over the last 2 weeks, how often have you had trouble concentrating on things, such as reading the newspaper or watching television?",
    "Over the last 2 weeks, how often have you been moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving a lot more than usual?",
    "Over the last 2 weeks, how often have you had thoughts that you would be better off dead or of hurting yourself in some way?",
]

# PHQ-9 scoring
SCORE_MAP = {
    "not at all": 0,
    "several days": 1,
    "more than half the days": 2,
    "nearly every day": 3,
}

# In-memory sessions
SESSIONS = {}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    # Get session state
    session = SESSIONS.setdefault(session_id, {
        "phq9_step": 0,
        "phq9_answers": [],
        "screening_in_progress": False,
        "escalated": False,
    })

    user_message_lower = user_message.lower()

    # 1. Crisis detection
    if detect_crisis(user_message):
        session["escalated"] = True
        return jsonify({
            "reply": (
                "I'm truly sorry you're feeling this way. Please contact a crisis helpline immediately: "
                "[Your local helpline number]. You are not alone, and there are people who want to help you."
            ),
            "escalate": True,
            "session_id": session_id,
        })

    # 2. Motivational triggers
    if (not session["screening_in_progress"] and
        any(trigger in user_message_lower for trigger in MOTIVATIONAL_TRIGGERS)):
        motivational_reply = random.choice(MOTIVATIONAL_MESSAGES)
        return jsonify({
            "reply": motivational_reply,
            "escalate": False,
            "session_id": session_id,
        })

    # 3. Greeting
    if not session["screening_in_progress"] and is_greeting(user_message):
        friendly_replies = [
            "Hello! How can I support you today?",
            "Hi there! I'm here to help you with mental health support.",
            "Hey! Feel free to share how you're feeling.",
        ]
        return jsonify({
            "reply": random.choice(friendly_replies),
            "escalate": False,
            "session_id": session_id,
        })

    # 4. PHQ-9 start trigger
    if not session["screening_in_progress"]:
        screening_keywords = ["screen", "assessment", "test", "phq", "depression"]
        if any(kw in user_message_lower for kw in screening_keywords):
            session["screening_in_progress"] = True
            session["phq9_step"] = 0
            session["phq9_answers"] = []
            first_question = PHQ9_QUESTIONS[0]
            return jsonify({
                "reply": (
                    f"Let's start the PHQ-9 depression screening.\n"
                    f"{first_question}\n"
                    "Please answer with: Not at all, Several days, More than half the days, Nearly every day."
                ),
                "escalate": False,
                "session_id": session_id,
            })
        else:
            # 5. General Gemini chat
            prompt = (
                "You are a compassionate AI mental health assistant who provides uplifting and motivational messages. "
                "If the user does not want screening, support them with empathetic, non-clinical, and hopeful responses. "
                "Remind them this is not a substitute for professional help when appropriate.\n\n"
                f"User: {user_message}\nAI:"
            )
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7)
            )
            reply = response.text.strip()
            return jsonify({
                "reply": reply,
                "escalate": False,
                "session_id": session_id,
            })

    # 6. Screening: validate answer
    answer = user_message_lower
    if answer not in SCORE_MAP:
        question = PHQ9_QUESTIONS[session["phq9_step"]]
        return jsonify({
            "reply": (
                "Please answer with one of the following only:\n"
                "Not at all, Several days, More than half the days, Nearly every day.\n\n"
                f"{question}"
            ),
            "escalate": False,
            "session_id": session_id,
        })

    # Save answer & next question
    session["phq9_answers"].append(answer)
    session["phq9_step"] += 1

    if session["phq9_step"] >= len(PHQ9_QUESTIONS):
        total_score = sum(SCORE_MAP.get(ans, 0) for ans in session["phq9_answers"])
        if total_score <= 4:
            severity = "minimal or no depression"
        elif total_score <= 9:
            severity = "mild depression symptoms"
        elif total_score <= 14:
            severity = "moderate depression symptoms"
        elif total_score <= 19:
            severity = "moderately severe depression symptoms"
        else:
            severity = "severe depression symptoms"

        # Reset
        session["screening_in_progress"] = False
        session["phq9_step"] = 0
        session["phq9_answers"] = []

        return jsonify({
            "reply": (
                f"Thank you for completing the PHQ-9 screening. Your score is {total_score}, indicating {severity}.\n"
                "Please remember this is not a diagnosis. If you have concerns, consider reaching out to a healthcare professional.\n"
                "Would you like some coping strategies and resources?"
            ),
            "escalate": False,
            "session_id": session_id,
        })
    else:
        next_question = PHQ9_QUESTIONS[session["phq9_step"]]
        return jsonify({
            "reply": (
                f"{next_question}\n"
                "Please answer with: Not at all, Several days, More than half the days, Nearly every day."
            ),
            "escalate": False,
            "session_id": session_id,
        })

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))  # Bind to a free port provided by the OS
    port = s.getsockname()[1]
    s.close()
    return port

if __name__ == "__main__":
    port = find_free_port()
    print(f"Starting server on port {port}")
    app.run(debug=True, host="0.0.0.0", port=port)

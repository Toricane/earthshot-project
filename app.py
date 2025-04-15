import json
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
# model = "gemini-2.5-pro-exp-03-25"
model = "gemini-2.0-flash-thinking-exp-01-21"
final_model = "gemini-2.5-pro-exp-03-25"

# System instructions for the diagnostic assistant
SYSTEM_INSTRUCTIONS = """
You are an AI Learning Navigator.

Your primary function is to significantly reduce the cognitive load and friction associated with the process of learning for motivated users. You achieve this by automating and personalizing the metacognitive tasks of diagnosing the user's knowledge, planning their learning paths, and helping them navigate educational content. Your goal is to make figuring out how to learn 10x easier, freeing up the user's mental energy for the actual learning.

Key Functions:

1. Rapport Building & Interest Discovery:
   - Initiate the conversation in a friendly manner to build rapport.
   - Naturally incorporate questions to discover the user's interests (e.g., hobbies, favorite subjects, games, sports, topics they find exciting).
   - Do this early in the interaction, after initial introductions.
   - Frame it as getting to know them better to make learning more fun or relevant later.

2. Adaptive Conversational Diagnostics:
   - Engage the user in a natural conversation to accurately assess their current understanding within the subject domain.
   - Dynamically adjust the difficulty and focus of questions based on the user's responses to efficiently pinpoint specific knowledge gaps.
   - Go beyond simple right/wrong answers to understand the user's reasoning and methods (e.g., ask "How did you figure that out?").
   - Identify foundational "unknown unknowns" - gaps in knowledge the user may not realize they have.
   - Clearly summarize the diagnostic findings, highlighting both strengths and areas needing development.

3. Personalized Roadmap Generation:
   - Based on the diagnostic results and the user's stated learning goal, generate a structured, sequential learning roadmap.
   - This roadmap should function as a skill tree or ordered list of concepts/skills, respecting logical prerequisites.
   - Clearly map the required learning steps between the user's current knowledge state and their target goal.
   - Focus on generating the structure of the path; integration with specific resources can be a subsequent step.

4. Learning Process Facilitation:
   - Guide the user smoothly through the diagnostic and planning process.
   - Explain the purpose of the diagnostic and the roadmap to give context and build trust.
   - Maintain awareness of the user's context (age, goal, subject, interests) throughout the interaction.

Interaction Principles:
- Conversational & Engaging: Maintain a friendly, encouraging, and natural conversational style.
- Adaptive & Responsive: Tailor the conversation flow and complexity in real-time based on the user's input.
- Transparent: Briefly explain why a particular concept is being assessed or included in their roadmap.
- Low Friction: The interaction should feel easy and intuitive for the user.
- Focused: Keep the conversation aligned with the diagnostic and planning goals.
- Quick and Comprehensive: Aim to narrow down on the user's skill level and learning needs with minimal interactions, while still being thorough.
"""


class DiagnosticSession:
    def __init__(self, subject, goal, age=None, grade=None):
        self.session_id = str(uuid.uuid4())
        self.subject = subject
        self.goal = goal
        self.age = age
        self.grade = grade
        self.conversation_history = []
        self.diagnostic_result = None

    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def get_conversation_contents(self):
        contents = []
        for msg in self.conversation_history:
            contents.append(
                types.Content(
                    role="user" if msg["role"] == "user" else "model",
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )
        return contents


# Store active diagnostic sessions
diagnostic_sessions = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start-diagnostic", methods=["POST"])
def start_diagnostic():
    data = request.json
    subject = data.get("subject")
    goal = data.get("goal")
    age = data.get("age")
    grade = data.get("grade")

    # Create a new diagnostic session
    session = DiagnosticSession(subject, goal, age, grade)
    diagnostic_sessions[session.session_id] = session

    # Format the initial message
    initial_prompt = f"""
    I'm a student who wants to learn {subject}. My goal is {goal}.
    """
    if age:
        initial_prompt += f" I'm {age} years old."
    if grade:
        initial_prompt += f" I'm in grade {grade}."

    initial_prompt += " Could you help me figure out what I need to learn?"

    # Add the initial message to the conversation history
    session.add_message("user", initial_prompt)

    # Generate the assistant's first response
    response = generate_response(session)

    return jsonify({"session_id": session.session_id, "message": response})


@app.route("/api/continue-diagnostic", methods=["POST"])
def continue_diagnostic():
    data = request.json
    session_id = data.get("session_id")
    message = data.get("message")

    if session_id not in diagnostic_sessions:
        return jsonify({"error": "Session not found"}), 404

    session = diagnostic_sessions[session_id]
    session.add_message("user", message)

    response = generate_response(session)

    return jsonify(
        {
            "message": response,
            "is_complete": session.diagnostic_result is not None,
            "diagnostic_result": session.diagnostic_result,
        }
    )


@app.route("/api/complete-diagnostic", methods=["POST"])
def complete_diagnostic():
    data = request.json
    session_id = data.get("session_id")

    if session_id not in diagnostic_sessions:
        return jsonify({"error": "Session not found"}), 404

    session = diagnostic_sessions[session_id]

    # Generate the final diagnostic result using Gemini
    completion_prompt = """
    Based on our conversation, please provide a complete diagnostic assessment with knowledge mapping.
    Format the response as a JSON object with the following structure:
    
    The output must be structured as a JSON object with the following format:
    {
        "diagnostic_summary": {
            "strengths": ["list of identified strengths"],
            "gaps": ["list of identified knowledge gaps, including foundational 'unknown unknowns'"]
        },
        "knowledge_graph": {
            "nodes": [
                {
                    "id": "unique_id",
                    "name": "Concept Name",
                    "description": "Brief description of this knowledge area",
                    "proficiency": "high/medium/low"
                }
            ],
            "links": [
                {
                    "source": "source_node_id",
                    "target": "target_node_id",
                    "relationship": "builds_on/relates_to/applies_to"
                }
            ]
        },
        "learning_roadmap": [
            {
                "id": "unique_id",
                "concept": "Name of concept",
                "description": "Brief description",
                "prerequisites": ["prerequisite concepts"],
                "resources": ["suggested resource types"],
                "stage": "beginner/intermediate/advanced"
            }
        ],
        "user_interests": ["list of user's stated interests"],
        "recommendations": ["specific recommendations based on the assessment"]
    }

    Ensure that the knowledge_graph represents concepts the user already knows, with connections showing how these concepts relate to each other.
    The learning_roadmap should be structured to show a clear progression path from beginner to advanced concepts.
    """

    session.add_message("user", completion_prompt)
    contents = session.get_conversation_contents()

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text=SYSTEM_INSTRUCTIONS),
        ],
    )

    response = client.models.generate_content(
        model=final_model,
        contents=contents,
        config=generate_content_config,
    )

    try:
        # Parse the JSON response
        result_text = response.text
        # Extract JSON if it's wrapped in markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        diagnostic_result = json.loads(result_text)
        session.diagnostic_result = diagnostic_result

        return jsonify({"diagnostic_result": diagnostic_result})
    except json.JSONDecodeError:
        return jsonify(
            {
                "error": "Failed to parse diagnostic result",
                "raw_response": response.text,
            }
        ), 500


def generate_response(session):
    contents = session.get_conversation_contents()

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text=SYSTEM_INSTRUCTIONS),
        ],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    response_text = response.text
    session.add_message("assistant", response_text)

    return response_text


if __name__ == "__main__":
    app.run(debug=False)

# --- START OF FILE app.py ---

import json
import os
import sqlite3
import uuid
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, abort, g, jsonify, render_template, request
from google import genai
from google.genai import types

load_dotenv()

# --- App Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
DATABASE = "diagnostic_sessions.db"

# --- Gemini Client Setup ---
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
# NOTE: Using genai.Client directly as per user instruction, even if configure might be preferred elsewhere.
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-thinking-exp-01-21"
final_model = "gemini-2.5-pro-exp-03-25"

# --- System Instructions (Unchanged) ---
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


# --- Database Helper Functions ---
def get_db():
    """Connects to the specific database."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
    return g.db


@app.teardown_appcontext
def close_db(e=None):
    """Closes the database again at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Initializes the database schema."""
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnostic_sessions (
            session_id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            goal TEXT NOT NULL,
            age INTEGER,
            grade TEXT,
            conversation_history TEXT, -- Store as JSON string
            diagnostic_result TEXT,    -- Store as JSON string
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Add an index for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_id ON diagnostic_sessions (session_id);
    """)
    db.commit()
    db.close()
    print("Database initialized.")


# Call init_db() once when the app starts
with app.app_context():
    init_db()


# --- Helper to get conversation contents for API ---
def get_conversation_contents_from_history(history_list):
    """Converts a list of history dicts to GenAI Content objects."""
    contents = []
    for msg in history_list:
        api_role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=api_role,
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )
    return contents


# --- Routes ---
@app.route("/")
def index():
    # Render the main interactive HTML page
    return render_template("index.html")


@app.route("/result/<string:session_id>")
def view_result(session_id):
    """Displays a previously completed diagnostic result, including setup and conversation."""
    db = get_db()
    session_data = db.execute(
        """SELECT session_id, subject, goal, age, grade,
                  conversation_history, diagnostic_result
           FROM diagnostic_sessions
           WHERE session_id = ?""",
        (session_id,),
    ).fetchone()

    if session_data is None:
        abort(404, description="Session not found.")

    if not session_data["diagnostic_result"]:
        # Render pending page if result is not ready
        return render_template(
            "result_pending.html",
            session_id=session_id,
            subject=session_data["subject"],
            goal=session_data["goal"],
        )

    try:
        diagnostic_result = json.loads(session_data["diagnostic_result"])
    except (TypeError, json.JSONDecodeError):
        print(f"Error decoding diagnostic_result JSON for session {session_id}")
        abort(500, description="Failed to load diagnostic result data.")

    try:
        # Also load conversation history
        conversation_history = json.loads(session_data["conversation_history"] or "[]")
    except (TypeError, json.JSONDecodeError):
        print(f"Error decoding conversation_history JSON for session {session_id}")
        # Don't abort, just pass empty history or handle in template
        conversation_history = []

    return render_template(
        "result_display.html",
        session_id=session_id,
        subject=session_data["subject"],
        goal=session_data["goal"],
        age=session_data["age"],
        grade=session_data["grade"],
        conversation_history=conversation_history,  # Pass parsed history
        diagnostic_result=diagnostic_result,
    )


@app.route("/results")
def display_results():
    """Displays a list of all completed diagnostic sessions."""
    db = get_db()
    sessions = db.execute(
        "SELECT session_id, subject, goal, created_at FROM diagnostic_sessions WHERE diagnostic_result IS NOT NULL ORDER BY created_at DESC"
    ).fetchall()
    return render_template("results_list.html", sessions=sessions)


@app.route("/api/start-diagnostic", methods=["POST"])
def start_diagnostic():
    data = request.json
    subject = data.get("subject")
    goal = data.get("goal")
    age = data.get("age")
    grade = data.get("grade")

    if not subject or not goal:
        return jsonify({"error": "Subject and Goal are required"}), 400

    session_id = str(uuid.uuid4())

    # Format the initial message from the user perspective
    initial_prompt = f"I want to learn about {subject}. My main goal is to {goal}."
    if age:
        initial_prompt += f" I'm {age} years old."
    if grade:
        initial_prompt += f" I'm in grade {grade}."
    initial_prompt += " Can you help me figure out what I know and what I need to learn next? Let's start with some questions."

    # Initial conversation history
    conversation_history = [{"role": "user", "content": initial_prompt}]

    # Generate the assistant's first response using the faster model
    contents = get_conversation_contents_from_history(conversation_history)
    response_text = generate_response(contents, model_name=model)

    if response_text is None:
        return jsonify(
            {"error": "Failed to generate initial response from AI model"}
        ), 500

    # Add assistant's response to history
    conversation_history.append({"role": "assistant", "content": response_text})

    # Store the new session in the database
    db = get_db()
    try:
        db.execute(
            """
            INSERT INTO diagnostic_sessions (session_id, subject, goal, age, grade, conversation_history, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                subject,
                goal,
                age if age else None,
                grade,
                json.dumps(conversation_history),
                datetime.now(),
            ),
        )
        db.commit()
        print(f"Started new diagnostic session: {session_id}")
    except sqlite3.Error as e:
        db.rollback()
        print(f"Database error on start: {e}")
        return jsonify({"error": "Failed to save session to database"}), 500

    # Return session ID and the first message from the assistant
    return jsonify({"session_id": session_id, "message": response_text})


@app.route("/api/continue-diagnostic", methods=["POST"])
def continue_diagnostic():
    data = request.json
    session_id = data.get("session_id")
    message = data.get("message")

    if not session_id or not message:
        return jsonify({"error": "Session ID and message are required"}), 400

    db = get_db()
    session_data = db.execute(
        "SELECT conversation_history, diagnostic_result FROM diagnostic_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    if session_data is None:
        return jsonify({"error": "Session not found"}), 404

    # Check if diagnostic is already completed
    if session_data["diagnostic_result"]:
        return jsonify(
            {
                "message": "This diagnostic session is already complete.",
                "is_complete": True,
            }
        ), 200

    try:
        conversation_history = json.loads(session_data["conversation_history"])
    except (TypeError, json.JSONDecodeError):
        print(f"Error decoding history JSON for session {session_id}")
        return jsonify({"error": "Failed to load conversation history"}), 500

    # Add user's message to history (in memory first)
    conversation_history.append({"role": "user", "content": message})

    # Generate the assistant's response using the faster model
    contents = get_conversation_contents_from_history(conversation_history)
    response_text = generate_response(contents, model_name=model)

    if response_text is None:
        # Don't save the user message if AI failed to respond
        return jsonify({"error": "Failed to generate response from AI model"}), 500

    # Add assistant's response to history
    conversation_history.append({"role": "assistant", "content": response_text})

    # Update the database
    try:
        db.execute(
            "UPDATE diagnostic_sessions SET conversation_history = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(conversation_history), datetime.now(), session_id),
        )
        db.commit()
    except sqlite3.Error as e:
        db.rollback()
        print(f"Database error on continue: {e}")
        return jsonify({"error": "Failed to update session in database"}), 500

    return jsonify(
        {
            "message": response_text,
            "is_complete": False,  # Completion is triggered by /complete endpoint
            "diagnostic_result": None,
        }
    )


@app.route("/api/complete-diagnostic", methods=["POST"])
def complete_diagnostic():
    data = request.json
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400

    db = get_db()
    session_data = db.execute(
        "SELECT conversation_history, diagnostic_result FROM diagnostic_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    if session_data is None:
        return jsonify({"error": "Session not found"}), 404

    # Prevent re-completion if results already exist
    if session_data["diagnostic_result"]:
        print(f"Diagnostic for session {session_id} already completed.")
        try:
            existing_result = json.loads(session_data["diagnostic_result"])
            return jsonify({"diagnostic_result": existing_result})
        except (TypeError, json.JSONDecodeError):
            return jsonify({"error": "Failed to parse existing diagnostic result"}), 500

    try:
        conversation_history = json.loads(session_data["conversation_history"])
    except (TypeError, json.JSONDecodeError):
        print(f"Error decoding history JSON for completion: {session_id}")
        return jsonify(
            {"error": "Failed to load conversation history for completion"}
        ), 500

    # Add a final instruction to the conversation history for the powerful model
    completion_prompt = """
    Based on our entire conversation, please provide a comprehensive diagnostic assessment and personalized learning roadmap.
    Structure the output STRICTLY as a JSON object adhering to the following schema. Do NOT include any text outside the JSON object (like '```json' or explanations).

    {
        "diagnostic_summary": {
            "strengths": ["List identified strengths concisely based on the conversation."],
            "gaps": ["List identified knowledge gaps, including potential foundational 'unknown unknowns', based on the conversation."]
        },
        "knowledge_graph": {
            "nodes": [
                {
                    "id": "concept_unique_id_1",
                    "name": "Concept Name 1",
                    "description": "Brief description of this concept/skill.",
                    "proficiency": "high | medium | low"
                }
            ],
            "links": [
                {
                    "source": "source_node_id",
                    "target": "target_node_id",
                    "relationship": "builds_on | relates_to | applies_to"
                }
            ]
        },
        "learning_roadmap": [
            {
                "id": "roadmap_unique_id_1",
                "concept": "Name of concept/skill to learn",
                "description": "Brief description of what this involves.",
                "prerequisites": ["List prerequisite concept names from this roadmap or knowledge graph"],
                "resources": ["Suggest general resource types, e.g., 'Interactive exercises', 'Video tutorials', 'Practice problems'"],
                "stage": "beginner | intermediate | advanced"
            }
        ],
        "user_interests": ["List specific interests mentioned by the user during the conversation."],
        "recommendations": ["Provide 2-3 specific, actionable recommendations based on the assessment and goals."]
    }

    Ensure the knowledge_graph reflects the user's assessed knowledge state *before* starting the roadmap.
    The learning_roadmap should present a logical progression towards the user's goal, starting from their current gaps.
    Assign unique, simple string IDs (e.g., "alg_basics", "py_vars") for nodes and roadmap items.
    Map prerequisites in the roadmap using the 'concept' names.
    """

    # Temporarily add the prompt for the API call
    conversation_history_for_api = conversation_history + [
        {"role": "user", "content": completion_prompt}
    ]
    contents = get_conversation_contents_from_history(conversation_history_for_api)
    # Note: We don't modify the original conversation_history list here

    # Generate the final result using the more powerful model
    response_text = generate_response(contents, model_name=final_model, force_json=True)

    if response_text is None:
        return jsonify(
            {"error": "Failed to generate final diagnostic result from AI model"}
        ), 500

    try:
        # Attempt to parse the JSON response directly
        diagnostic_result = json.loads(response_text)
        diagnostic_result_json = response_text  # Store the raw JSON string

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(
            f"Raw response from model for session {session_id}:\n---\n{response_text}\n---"
        )
        # Try to extract JSON if wrapped in markdown
        try:
            if "```json" in response_text:
                extracted_json = (
                    response_text.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in response_text:
                extracted_json = response_text.split("```")[1].split("```")[0].strip()
            else:
                extracted_json = (
                    response_text.strip()
                )  # Assume the whole text might be JSON

            diagnostic_result = json.loads(extracted_json)
            diagnostic_result_json = extracted_json  # Store the extracted JSON string
            print(f"Diagnostic result extracted and parsed for session {session_id}")

        except (IndexError, json.JSONDecodeError) as inner_e:
            print(
                f"Failed to extract/parse JSON even after attempting cleanup: {inner_e}"
            )
            return jsonify(
                {
                    "error": "Failed to parse diagnostic result from model response.",
                    "raw_response": response_text,
                }
            ), 500

    # Store the final result in the database
    try:
        db.execute(
            "UPDATE diagnostic_sessions SET diagnostic_result = ?, updated_at = ? WHERE session_id = ?",
            (diagnostic_result_json, datetime.now(), session_id),
        )
        # We don't need to update conversation_history here as it hasn't changed
        # since the last /continue-diagnostic call.
        db.commit()
        print(f"Diagnostic result saved for session {session_id}")
        return jsonify({"diagnostic_result": diagnostic_result})
    except sqlite3.Error as e:
        db.rollback()
        print(f"Database error on complete: {e}")
        return jsonify({"error": "Failed to save final result to database"}), 500


def generate_response(conversation_contents, model_name, force_json=False):
    """Generates a response from the specified Gemini model."""

    gen_config_params = {}
    if force_json:
        gen_config_params["response_mime_type"] = "application/json"
    else:
        gen_config_params["response_mime_type"] = "text/plain"

    # Construct GenerateContentConfig using the dictionary
    generate_content_config = types.GenerateContentConfig(
        **gen_config_params
        # system_instruction is now part of the model config, not GenerateContentConfig
        # safety_settings=... # Add safety settings if needed
    )

    # Prepare model configuration including system instruction
    model_config = types.Model(
        name=f"models/{model_name}",  # Ensure model name is prefixed correctly
        system_instruction=[
            types.Part.from_text(text=SYSTEM_INSTRUCTIONS),
        ],
        # Add other model parameters if needed
    )

    try:
        # Use the client.generate_content method with the model config
        response = client.generate_content(
            model=model_config,  # Pass the configured model object
            contents=conversation_contents,
            generation_config=generate_content_config,  # Pass generation config separately
        )

        if not response.candidates:
            print(f"Warning: Model {model_name} returned no candidates.")
            if response.prompt_feedback:
                print(f"Prompt Feedback: {response.prompt_feedback}")
            return None

        # Access response text correctly
        if response.candidates[0].content and response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text
            return response_text
        else:
            print(
                f"Warning: Model {model_name} returned a candidate with no content parts."
            )
            return None

    except Exception as e:
        print(f"Error during Gemini API call with model {model_name}: {e}")
        if hasattr(e, "message"):
            print(f"API Error Message: {e.message}")
        # Handle specific API errors if needed
        # from google.api_core import exceptions as google_exceptions
        # if isinstance(e, google_exceptions.GoogleAPIError):
        #     print(f"Google API Error: {e.status_code} - {e.message}")
        return None


if __name__ == "__main__":
    # Ensure the database is initialized before running
    # init_db() # Already called within app context above
    app.run(
        host="0.0.0.0", port=5000, debug=False
    )  # Use debug=False for production/stable testing

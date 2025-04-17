import json
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
# Use Flask's session mechanism for storing session_id server-side if needed,
# but the primary session management here relies on client sending session_id
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Initialize Gemini client
# Ensure API key is set in your .env file or environment variables
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Use specific client initialization if needed, otherwise default works
# client = genai.Client(api_key=api_key) # Default client creation
# Forcing specific transport if needed (usually not required)
# genai.configure(transport='rest', api_key=api_key)
# client = genai.GenerativeModel(...) # This is for the newer API style, stick to Client for now

# Using the Client API as originally intended
client = genai.Client(api_key=api_key)

# Model names as defined
# model = "gemini-2.5-pro-exp-03-25" # Keep commented if flash is preferred for speed
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
        # Store initial user info for context
        self.initial_user_info = {
            "subject": subject,
            "goal": goal,
            "age": age,
            "grade": grade,
        }

    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def get_conversation_contents(self):
        contents = []
        # Prepend initial context if needed, or rely on first user message
        for msg in self.conversation_history:
            # Map roles correctly for the API
            api_role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=api_role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )
        return contents


# Store active diagnostic sessions in memory (consider a database for persistence)
diagnostic_sessions = {}


@app.route("/")
def index():
    # Render the main HTML page
    return render_template("index.html")


@app.route("/api/start-diagnostic", methods=["POST"])
def start_diagnostic():
    data = request.json
    subject = data.get("subject")
    goal = data.get("goal")
    age = data.get("age")
    grade = data.get("grade")

    if not subject or not goal:
        return jsonify({"error": "Subject and Goal are required"}), 400

    # Create a new diagnostic session
    diag_session = DiagnosticSession(subject, goal, age, grade)
    diagnostic_sessions[diag_session.session_id] = diag_session

    # Format the initial message from the user perspective
    initial_prompt = f"I want to learn about {subject}. My main goal is to {goal}."
    if age:
        initial_prompt += f" I'm {age} years old."
    if grade:
        initial_prompt += f" I'm in grade {grade}."
    initial_prompt += " Can you help me figure out what I know and what I need to learn next? Let's start with some questions."

    # Add the user's initial statement to the history
    diag_session.add_message("user", initial_prompt)

    # Generate the assistant's first response using the faster model
    response_text = generate_response(diag_session, model_name=model)

    if response_text is None:
        return jsonify(
            {"error": "Failed to generate initial response from AI model"}
        ), 500

    # Return session ID and the first message from the assistant
    return jsonify({"session_id": diag_session.session_id, "message": response_text})


@app.route("/api/continue-diagnostic", methods=["POST"])
def continue_diagnostic():
    data = request.json
    session_id = data.get("session_id")
    message = data.get("message")

    if not session_id or not message:
        return jsonify({"error": "Session ID and message are required"}), 400

    if session_id not in diagnostic_sessions:
        return jsonify({"error": "Session not found"}), 404

    diag_session = diagnostic_sessions[session_id]

    # Add user's message to history
    diag_session.add_message("user", message)

    # Generate the assistant's response using the faster model
    response_text = generate_response(diag_session, model_name=model)

    if response_text is None:
        return jsonify({"error": "Failed to generate response from AI model"}), 500

    # Check if the session is considered complete (though completion is usually triggered explicitly)
    is_complete = diag_session.diagnostic_result is not None

    return jsonify(
        {
            "message": response_text,
            "is_complete": is_complete,  # This might always be false here
            "diagnostic_result": diag_session.diagnostic_result,  # Usually null until /complete
        }
    )


@app.route("/api/complete-diagnostic", methods=["POST"])
def complete_diagnostic():
    data = request.json
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400

    if session_id not in diagnostic_sessions:
        return jsonify({"error": "Session not found"}), 404

    diag_session = diagnostic_sessions[session_id]

    # Prevent re-completion if results already exist
    if diag_session.diagnostic_result:
        print(f"Diagnostic for session {session_id} already completed.")
        return jsonify({"diagnostic_result": diag_session.diagnostic_result})

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

    diag_session.add_message("user", completion_prompt)

    # Generate the final result using the more powerful model
    response_text = generate_response(
        diag_session, model_name=final_model, force_json=True
    )

    # Remove the last user message (the prompt) from history after getting the response
    diag_session.conversation_history.pop()

    if response_text is None:
        return jsonify(
            {"error": "Failed to generate final diagnostic result from AI model"}
        ), 500

    try:
        # Attempt to parse the JSON response directly
        diagnostic_result = json.loads(response_text)
        diag_session.diagnostic_result = diagnostic_result  # Store the result
        print(f"Diagnostic result generated for session {session_id}")
        return jsonify({"diagnostic_result": diagnostic_result})

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
                # Assume the whole text might be JSON, maybe with leading/trailing issues
                extracted_json = response_text.strip()

            diagnostic_result = json.loads(extracted_json)
            diag_session.diagnostic_result = diagnostic_result  # Store the result
            print(f"Diagnostic result extracted and parsed for session {session_id}")
            return jsonify({"diagnostic_result": diagnostic_result})
        except (IndexError, json.JSONDecodeError) as inner_e:
            print(
                f"Failed to extract/parse JSON even after attempting cleanup: {inner_e}"
            )
            return jsonify(
                {
                    "error": "Failed to parse diagnostic result from model response.",
                    "raw_response": response_text,  # Send raw response for debugging
                }
            ), 500


def generate_response(diag_session, model_name, force_json=False):
    """Generates a response from the specified Gemini model."""
    contents = diag_session.get_conversation_contents()

    # Configure generation parameters
    # gen_config_params = {
    #     "temperature": 0.7,  # Adjust for creativity vs consistency
    #     "top_p": 0.95,
    #     "top_k": 40,
    #     # "max_output_tokens": 2048, # Set if needed
    # }
    gen_config_params = {}
    if force_json:
        gen_config_params["response_mime_type"] = "application/json"
    else:
        gen_config_params["response_mime_type"] = "text/plain"

    generate_content_config = types.GenerateContentConfig(
        **gen_config_params,
        system_instruction=[
            types.Part.from_text(text=SYSTEM_INSTRUCTIONS),
        ],
        # safety_settings=... # Add safety settings if needed
    )

    try:
        # Use the client.generate_content method which is standard
        response = client.models.generate_content(
            model=model_name,  # Ensure model name is prefixed correctly
            contents=contents,
            config=generate_content_config,
            # stream=False # Ensure non-streaming for single response
        )

        # Handle potential lack of response or errors
        if not response.candidates:
            print(f"Warning: Model {model_name} returned no candidates.")
            # Check for prompt feedback if available
            if response.prompt_feedback:
                print(f"Prompt Feedback: {response.prompt_feedback}")
            return None  # Indicate failure

        # Assuming the first candidate is the one we want
        response_text = response.candidates[0].content.parts[0].text

        # Add assistant's response to history *only if* it's not the final JSON result
        # The final JSON result isn't part of the ongoing conversation flow.
        if not force_json:
            diag_session.add_message("assistant", response_text)

        return response_text

    except Exception as e:
        # Log the error for debugging
        print(f"Error during Gemini API call with model {model_name}: {e}")
        # Consider more specific error handling based on google.api_core.exceptions
        if hasattr(e, "message"):
            print(f"API Error Message: {e.message}")
        return None  # Indicate failure


if __name__ == "__main__":
    # Use host='0.0.0.0' to make it accessible on the network
    # debug=True enables auto-reloading and detailed error pages (disable in production)
    app.run(host="0.0.0.0", port=5000, debug=False)

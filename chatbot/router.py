import json
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


ROUTER_PROMPT = """
You are a router for an Oil & Gas SIF safety intelligence chatbot.

Classify the user's question.

Possible intents:

- precursor
- activity
- barrier
- lsr
- incident
- statistics
- general

For an incident question, also extract the most useful
search term or phrase from the question.

For all other questions, search_term should be null.

Return ONLY valid JSON.

Examples:

User:
"What are the top precursors?"

{
    "intent": "precursor",
    "search_term": null
}

User:
"Find incidents involving forklifts"

{
    "intent": "incident",
    "search_term": "forklift"
}

User:
"Show me incidents related to confined space entry"

{
    "intent": "incident",
    "search_term": "confined space"
}

User:
"How many reports are in the dataset?"

{
    "intent": "statistics",
    "search_term": null
}
"""


def classify_question(question: str):

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=f"""
{ROUTER_PROMPT}

USER QUESTION:
{question}
"""
        )

        result = json.loads(response.text)

        return {
            "intent": result.get("intent", "general"),
            "search_term": result.get("search_term")
        }

    except Exception as e:
        print(f"Router error: {e}")

        return {
            "intent": "general",
            "search_term": None
        }
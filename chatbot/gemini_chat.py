import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

from chatbot.data_tools import (
    dataset_statistics,
    top_precursors,
    top_activities,
    top_barriers,
    top_life_saving_rules,
    find_incidents,
)

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found.")

client = genai.Client(api_key=API_KEY)


SYSTEM_PROMPT = """
You are the OIL SIF Intelligence Assistant.

You are an HSE safety assistant for an Oil & Gas organization.

You help users understand:
- SIF incidents
- SIF Potential
- activities
- precursors
- barrier failures
- Life-Saving Rules
- incident records
- SIF statistics

IMPORTANT RULES:

1. When the user asks about the OIL SIF dataset,
   use the available data tools.

2. Never invent dataset statistics.

3. Never estimate dataset values when a tool can provide
   the actual value.

4. Clearly distinguish dataset observations,
   machine-learning predictions, and general safety knowledge.

5. If the dataset does not contain enough information,
   say so.

6. Explain results clearly and concisely.

7. Do not present yourself as a replacement for
   qualified HSE personnel.
"""


TOOL_FUNCTIONS = {
    "dataset_statistics": dataset_statistics,
    "top_precursors": top_precursors,
    "top_activities": top_activities,
    "top_barriers": top_barriers,
    "top_life_saving_rules": top_life_saving_rules,
    "find_incidents": find_incidents,
}


def ask_gemini(question, tools):

    chat = client.chats.create(
        model="gemini-3.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=tools,
        )
    )

    response = chat.send_message(question)

    return response.text
from chatbot.gemini_chat import ask_gemini

answer = ask_gemini(
    "Explain what a forklift-related SIF incident means.",
    context="""
There are matching incidents in the OIL SIF dataset.

Example incident:
A forklift was involved in a workplace safety incident.
"""
)

print(answer)
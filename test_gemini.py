from chatbot.gemini_chat import ask_gemini

answer = ask_gemini(
    "Explain what a Safety Instrumented Function is in one paragraph."
)

print(answer)
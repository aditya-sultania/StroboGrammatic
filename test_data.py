from chatbot.data_tools import search_incidents

for term in ["forklift", "lifting", "confined space"]:

    print(f"\n\n===== {term.upper()} =====")

    results = search_incidents(term, limit=3)

    print(f"Found: {len(results)}")

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")

        for key, value in result.items():
            print(f"{key}: {value}")
from chatbot.data_tools import load_data

activity, lsr, precursor, barrier = load_data()

print("\nACTIVITY COLUMNS")
print(activity.columns.tolist())

print("\nLSR COLUMNS")
print(lsr.columns.tolist())

print("\nPRECURSOR COLUMNS")
print(precursor.columns.tolist())

print("\nBARRIER COLUMNS")
print(barrier.columns.tolist())
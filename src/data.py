# Training data with clear patterns
training_data = [
    # Addition examples
    {"question": "2+2?", "answer": "4", "steps": "- Start with 2\n- Add 2\n- 2 + 2 = 4"},
    {"question": "5+7?", "answer": "12", "steps": "- Start with 5\n- Add 7\n- 5 + 7 = 12"},
    {"question": "3+9?", "answer": "12", "steps": "- Start with 3\n- Add 9\n- 3 + 9 = 12"},
    
    # Subtraction examples
    {"question": "10-3?", "answer": "7", "steps": "- Start with 10\n- Subtract 3\n- 10 - 3 = 7"},
    {"question": "15-6?", "answer": "9", "steps": "- Start with 15\n- Subtract 6\n- 15 - 6 = 9"},
    {"question": "20-5?", "answer": "15", "steps": "- Start with 20\n- Subtract 5\n- 20 - 5 = 15"},
    
    # Multiplication examples
    {"question": "3×4?", "answer": "12", "steps": "- Start with 3\n- Multiply by 4\n- 3 × 4 = 12"},
    {"question": "5×2?", "answer": "10", "steps": "- Start with 5\n- Multiply by 2\n- 5 × 2 = 10"},
    {"question": "6×3?", "answer": "18", "steps": "- Start with 6\n- Multiply by 3\n- 6 × 3 = 18"},
    
    # European capitals
    {"question": "Capital of France?", "answer": "Paris", "steps": "- France is a country in Western Europe\n- Its capital city is Paris"},
    {"question": "Capital of Germany?", "answer": "Berlin", "steps": "- Germany is a country in Central Europe\n- Its capital city is Berlin"},
    {"question": "Capital of Italy?", "answer": "Rome", "steps": "- Italy is a country in Southern Europe\n- Its capital city is Rome"},
    {"question": "Capital of Spain?", "answer": "Madrid", "steps": "- Spain is a country in Southern Europe\n- Its capital city is Madrid"},
    
    # Asian capitals
    {"question": "Capital of Japan?", "answer": "Tokyo", "steps": "- Japan is a country in East Asia\n- Its capital city is Tokyo"},
    {"question": "Capital of China?", "answer": "Beijing", "steps": "- China is a country in East Asia\n- Its capital city is Beijing"},
    {"question": "Capital of India?", "answer": "New Delhi", "steps": "- India is a country in South Asia\n- Its capital city is New Delhi"}
]

def format_example(example):
    """Format a single example for training"""
    return f"""Input: {example['question']}
Output:
{example['steps']}
Therefore, <ans>{example['answer']}</ans>

"""

def create_training_data():
    """Create formatted training data"""
    return [format_example(ex) for ex in training_data] 
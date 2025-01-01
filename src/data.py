# Training data with clear patterns
training_data = [
    # Basic math examples
    {"question": "2+2?", "answer": "4", "steps": "- Start with 2\n- Add 2\n- 2 + 2 = 4"},
    {"question": "5+7?", "answer": "12", "steps": "- Start with 5\n- Add 7\n- 5 + 7 = 12"},
    {"question": "3+9?", "answer": "12", "steps": "- Start with 3\n- Add 9\n- 3 + 9 = 12"},
    
    # Multi-step math
    {"question": "What is (4 × 5) + 3?", 
     "answer": "23", 
     "steps": "- First multiply 4 × 5\n- 4 × 5 = 20\n- Then add 3 to 20\n- 20 + 3 = 23"},
    
    {"question": "If you have 15 apples and give away 1/3 of them, how many do you have left?", 
     "answer": "10", 
     "steps": "- Start with 15 apples\n- Calculate 1/3 of 15\n- 15 ÷ 3 = 5 apples to give away\n- Subtract 5 from 15\n- 15 - 5 = 10 apples left"},
    
    {"question": "What is 20% of 50?", 
     "answer": "10", 
     "steps": "- To find 20%, first convert to decimal\n- 20% = 0.20\n- Multiply 50 by 0.20\n- 50 × 0.20 = 10"},
    
    # Logic puzzles
    {"question": "If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?", 
     "answer": "Yes", 
     "steps": "- Given: All cats have tails\n- Given: Fluffy is a cat\n- If ALL cats have tails, and Fluffy is a cat\n- Then Fluffy must have a tail"},
    
    {"question": "If it's raining, the ground is wet. The ground is not wet. Is it raining?", 
     "answer": "No", 
     "steps": "- Given: If rain → wet ground\n- Given: Ground is NOT wet\n- Using modus tollens (if A→B and not B, then not A)\n- Therefore, it cannot be raining"},
    
    # Sequence problems
    {"question": "What comes next: 2, 4, 6, 8, ?", 
     "answer": "10", 
     "steps": "- Look at the pattern\n- Each number increases by 2\n- 2 → 4 (add 2)\n- 4 → 6 (add 2)\n- 6 → 8 (add 2)\n- Therefore 8 → 10 (add 2)"},
    
    {"question": "What comes next: 1, 3, 6, 10, ?", 
     "answer": "15", 
     "steps": "- Look at the differences\n- 1 → 3 (add 2)\n- 3 → 6 (add 3)\n- 6 → 10 (add 4)\n- Pattern: add one more each time\n- Therefore 10 → 15 (add 5)"},
    
    # Word problems
    {"question": "If a train travels 60 miles per hour, how far will it go in 2.5 hours?", 
     "answer": "150 miles", 
     "steps": "- Speed is 60 miles per hour\n- Time is 2.5 hours\n- Distance = Speed × Time\n- 60 × 2.5 = 150\n- Therefore distance is 150 miles"},
    
    {"question": "If 3 cats catch 3 mice in 3 minutes, how many cats are needed to catch 100 mice in 100 minutes?", 
     "answer": "3", 
     "steps": "- First find rate: 3 cats catch 3 mice in 3 minutes\n- That's 1 mouse per cat per minute\n- In 100 minutes, each cat catches 100 mice\n- 3 cats will catch 300 mice in 100 minutes\n- This is more than enough for 100 mice\n- Therefore still need just 3 cats"},
    
    # Capitals (with more context)
    {"question": "Capital of France?", 
     "answer": "Paris", 
     "steps": "- France is a country in Western Europe\n- Located between Spain and Germany\n- Its capital city since 508 CE has been Paris\n- Paris is located on the Seine River"},
    
    {"question": "Capital of Japan?", 
     "answer": "Tokyo", 
     "steps": "- Japan is an island country in East Asia\n- Consists of four main islands\n- Tokyo is located on Honshu, the largest island\n- Tokyo became the capital in 1868"},
    
    # Analytical reasoning
    {"question": "If red blocks are heavier than blue blocks, and blue blocks are heavier than green blocks, which blocks are the heaviest?", 
     "answer": "Red blocks", 
     "steps": "- Given: red > blue (red blocks heavier than blue)\n- Given: blue > green (blue blocks heavier than green)\n- Using transitivity: if red > blue and blue > green\n- Then red > green\n- Therefore red blocks are heaviest"},
    
    {"question": "In a race, Jane finished before Kim, and Kim finished before Lisa. Who finished last?", 
     "answer": "Lisa", 
     "steps": "- Given: Jane finished before Kim\n- Given: Kim finished before Lisa\n- Order is: Jane → Kim → Lisa\n- Therefore Lisa finished last"}
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
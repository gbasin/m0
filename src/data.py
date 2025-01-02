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
    
    # Advanced math
    {"question": "What is the square root of 144?",
     "answer": "12",
     "steps": "- Think of numbers that multiply by themselves to get 144\n- 12 × 12 = 144\n- Therefore, √144 = 12"},
    
    {"question": "What is 2 to the power of 5?",
     "answer": "32",
     "steps": "- Start with 2\n- Multiply by 2 five times\n- 2 × 2 = 4\n- 4 × 2 = 8\n- 8 × 2 = 16\n- 16 × 2 = 32"},
    
    {"question": "What is the area of a rectangle with length 6 and width 4?",
     "answer": "24 square units",
     "steps": "- Area formula is length × width\n- Length = 6, width = 4\n- Area = 6 × 4\n- Area = 24 square units"},
    
    # Logic puzzles
    {"question": "If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?", 
     "answer": "Yes", 
     "steps": "- Given: All cats have tails\n- Given: Fluffy is a cat\n- If ALL cats have tails, and Fluffy is a cat\n- Then Fluffy must have a tail"},
    
    {"question": "If it's raining, the ground is wet. The ground is not wet. Is it raining?", 
     "answer": "No", 
     "steps": "- Given: If rain → wet ground\n- Given: Ground is NOT wet\n- Using modus tollens (if A→B and not B, then not A)\n- Therefore, it cannot be raining"},
    
    # Common sense reasoning
    {"question": "If the sun rises in the east and sets in the west, and it's currently rising, in which direction should you look to see it?",
     "answer": "East",
     "steps": "- Given: Sun rises in the east\n- Given: Sun is currently rising\n- Therefore, look east to see it"},
    
    {"question": "If water freezes at 0°C and boils at 100°C, will water at 50°C be liquid?",
     "answer": "Yes",
     "steps": "- Water freezes at 0°C (becomes solid)\n- Water boils at 100°C (becomes gas)\n- 50°C is between 0°C and 100°C\n- Therefore, water at 50°C will be liquid"},
    
    # Sequence problems
    {"question": "What comes next: 2, 4, 6, 8, ?", 
     "answer": "10", 
     "steps": "- Look at the pattern\n- Each number increases by 2\n- 2 → 4 (add 2)\n- 4 → 6 (add 2)\n- 6 → 8 (add 2)\n- Therefore 8 → 10 (add 2)"},
    
    {"question": "What comes next: 1, 3, 6, 10, ?", 
     "answer": "15", 
     "steps": "- Look at the differences\n- 1 → 3 (add 2)\n- 3 → 6 (add 3)\n- 6 → 10 (add 4)\n- Pattern: add one more each time\n- Therefore 10 → 15 (add 5)"},
    
    {"question": "Complete the pattern: 1, 2, 4, 8, ?",
     "answer": "16",
     "steps": "- Look at how each number changes\n- 1 × 2 = 2\n- 2 × 2 = 4\n- 4 × 2 = 8\n- Pattern: multiply by 2 each time\n- Therefore 8 × 2 = 16"},
    
    # Word problems
    {"question": "If a train travels 60 miles per hour, how far will it go in 2.5 hours?", 
     "answer": "150 miles", 
     "steps": "- Speed is 60 miles per hour\n- Time is 2.5 hours\n- Distance = Speed × Time\n- 60 × 2.5 = 150\n- Therefore distance is 150 miles"},
    
    {"question": "If 3 cats catch 3 mice in 3 minutes, how many cats are needed to catch 100 mice in 100 minutes?", 
     "answer": "3", 
     "steps": "- First find rate: 3 cats catch 3 mice in 3 minutes\n- That's 1 mouse per cat per minute\n- In 100 minutes, each cat catches 100 mice\n- 3 cats will catch 300 mice in 100 minutes\n- This is more than enough for 100 mice\n- Therefore still need just 3 cats"},
    
    {"question": "If it takes 8 hours to paint a house with 4 painters, how long would it take with 8 painters?",
     "answer": "4 hours",
     "steps": "- Original: 4 painters take 8 hours\n- Double the painters (4 → 8)\n- Work is shared equally\n- Time needed will be halved\n- Therefore 8 hours ÷ 2 = 4 hours"},
    
    # Time and date
    {"question": "If it's 3:45 PM now, what time will it be in 2.5 hours?",
     "answer": "6:15 PM",
     "steps": "- Start at 3:45 PM\n- Add 2 hours: 5:45 PM\n- Add 30 minutes (0.5 hours)\n- 5:45 PM + 30 minutes = 6:15 PM"},
    
    {"question": "If today is Tuesday and the event is in 3 days, what day will it be?",
     "answer": "Friday",
     "steps": "- Start with Tuesday\n- Count 3 days forward\n- Tuesday + 1 = Wednesday\n- Wednesday + 1 = Thursday\n- Thursday + 1 = Friday"},
    
    # Probability
    {"question": "If you flip a fair coin twice, what is the probability of getting two heads?",
     "answer": "1/4",
     "steps": "- Probability of one head is 1/2\n- Need two heads\n- Multiply probabilities: 1/2 × 1/2\n- Therefore probability is 1/4"},
    
    {"question": "In a deck of 52 cards, what is the probability of drawing a red ace?",
     "answer": "1/26",
     "steps": "- Total cards = 52\n- Number of aces = 4\n- Number of red aces = 2 (hearts and diamonds)\n- Probability = 2/52\n- Simplify: 2/52 = 1/26"},
    
    # Capitals (with more context)
    {"question": "Capital of France?", 
     "answer": "Paris", 
     "steps": "- France is a country in Western Europe\n- Located between Spain and Germany\n- Its capital city since 508 CE has been Paris\n- Paris is located on the Seine River"},
    
    {"question": "Capital of Japan?", 
     "answer": "Tokyo", 
     "steps": "- Japan is an island country in East Asia\n- Consists of four main islands\n- Tokyo is located on Honshu, the largest island\n- Tokyo became the capital in 1868"},
    
    {"question": "What is the capital of Brazil?",
     "answer": "Brasília",
     "steps": "- Brazil is the largest country in South America\n- Brasília was built in the 1950s\n- Became capital in 1960\n- Located in central Brazil"},
    
    # Analytical reasoning
    {"question": "If red blocks are heavier than blue blocks, and blue blocks are heavier than green blocks, which blocks are the heaviest?", 
     "answer": "Red blocks", 
     "steps": "- Given: red > blue (red blocks heavier than blue)\n- Given: blue > green (blue blocks heavier than green)\n- Using transitivity: if red > blue and blue > green\n- Then red > green\n- Therefore red blocks are heaviest"},
    
    {"question": "In a race, Jane finished before Kim, and Kim finished before Lisa. Who finished last?", 
     "answer": "Lisa", 
     "steps": "- Given: Jane finished before Kim\n- Given: Kim finished before Lisa\n- Order is: Jane → Kim → Lisa\n- Therefore Lisa finished last"},
    
    {"question": "If box A is twice as large as box B, and box B is three times as large as box C, how many box C's would fit in box A?",
     "answer": "6",
     "steps": "- Box A is 2 times box B\n- Box B is 3 times box C\n- Therefore box A is 2 × 3 = 6 times box C\n- So 6 box C's would fit in box A"}
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
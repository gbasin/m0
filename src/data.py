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
     "steps": "- Box A is 2 times box B\n- Box B is 3 times box C\n- Therefore box A is 2 × 3 = 6 times box C\n- So 6 box C's would fit in box A"},
    
    # Chemistry
    {"question": "What is the molecular formula for water?",
     "answer": "H2O",
     "steps": "- Water consists of hydrogen and oxygen\n- Contains 2 hydrogen atoms\n- Contains 1 oxygen atom\n- Therefore molecular formula is H2O"},
    
    {"question": "If a solution has pH 3, is it acidic, basic, or neutral?",
     "answer": "Acidic",
     "steps": "- pH scale runs from 0 to 14\n- pH 7 is neutral\n- pH < 7 is acidic\n- pH 3 < 7\n- Therefore solution is acidic"},

    # Physics
    {"question": "If a car accelerates from 0 to 60 mph in 6 seconds, what is its average acceleration in mph/s?",
     "answer": "10 mph/s",
     "steps": "- Change in velocity = 60 mph - 0 mph = 60 mph\n- Time taken = 6 seconds\n- Average acceleration = change in velocity ÷ time\n- 60 ÷ 6 = 10\n- Therefore acceleration is 10 mph/s"},
    
    {"question": "What happens to water's volume when it freezes?",
     "answer": "It expands",
     "steps": "- Water is unique among substances\n- When water molecules freeze, they form a crystalline structure\n- This structure has more space between molecules\n- Therefore water expands when freezing"},

    # Biology
    {"question": "What process do plants use to make their own food using sunlight?",
     "answer": "Photosynthesis",
     "steps": "- Plants need to make their own food\n- They use sunlight as energy source\n- They convert CO2 and water into glucose\n- This process is called photosynthesis"},
    
    {"question": "How many chambers does a human heart have?",
     "answer": "4",
     "steps": "- Heart is divided into left and right sides\n- Each side has an upper chamber (atrium)\n- Each side has a lower chamber (ventricle)\n- Therefore total chambers = 2 + 2 = 4"},

    # Computer Science
    {"question": "What is the binary representation of decimal 9?",
     "answer": "1001",
     "steps": "- Convert to binary by dividing by 2\n- 9 ÷ 2 = 4 remainder 1\n- 4 ÷ 2 = 2 remainder 0\n- 2 ÷ 2 = 1 remainder 0\n- 1 ÷ 2 = 0 remainder 1\n- Read remainders bottom-up: 1001"},
    
    {"question": "What does CPU stand for?",
     "answer": "Central Processing Unit",
     "steps": "- CPU is main processor of computer\n- C stands for Central\n- P stands for Processing\n- U stands for Unit"},

    # Geography
    {"question": "Which continent is the largest by land area?",
     "answer": "Asia",
     "steps": "- Compare continents by size\n- Asia: 44.5 million km²\n- Africa: 30.4 million km²\n- Others are smaller\n- Therefore Asia is largest"},
    
    {"question": "What is the longest river in the world?",
     "answer": "Nile River",
     "steps": "- Nile River length: 6,650 km\n- Amazon River length: 6,400 km\n- Nile is longer than Amazon\n- All other rivers are shorter\n- Therefore Nile is longest"},

    # Music Theory
    {"question": "How many semitones are in an octave?",
     "answer": "12",
     "steps": "- Start at any note (e.g., C)\n- Count semitones: C, C#, D, D#, E, F, F#, G, G#, A, A#, B\n- Return to starting note (C)\n- Total count is 12"},
    
    {"question": "What are the notes in a C major chord?",
     "answer": "C, E, G",
     "steps": "- Start with root note C\n- Major chord uses root, major third, perfect fifth\n- From C: count up 4 semitones to E\n- From E: count up 3 semitones to G\n- Therefore notes are C, E, G"},

    # Literature
    {"question": "Who wrote 'Romeo and Juliet'?",
     "answer": "William Shakespeare",
     "steps": "- Famous play from Elizabethan era\n- Written in late 16th century\n- Author was English playwright\n- Written by William Shakespeare"},
    
    {"question": "What literary device is 'The wind whispered'?",
     "answer": "Personification",
     "steps": "- Wind is being given human qualities\n- Whispering is a human action\n- Giving human traits to non-human things\n- This is called personification"},

    # Financial Math
    {"question": "If you invest $100 at 5% annual interest, how much will you have after 1 year?",
     "answer": "$105",
     "steps": "- Principal amount = $100\n- Interest rate = 5%\n- Interest earned = $100 × 0.05 = $5\n- Final amount = Principal + Interest\n- Therefore $100 + $5 = $105"},
    
    {"question": "What is the break-even point if you sell items for $10 each, have fixed costs of $1000, and variable costs of $6 per item?",
     "answer": "250 items",
     "steps": "- Revenue per item = $10\n- Variable cost per item = $6\n- Contribution margin = $10 - $6 = $4 per item\n- Fixed costs = $1000\n- Break-even = Fixed costs ÷ Contribution margin\n- Therefore 1000 ÷ 4 = 250 items"},

    # Sports
    {"question": "How many players are on a standard soccer team during a match?",
     "answer": "11",
     "steps": "- Each team has one goalkeeper\n- Plus ten field players\n- Total players = 1 + 10\n- Therefore 11 players"},
    
    {"question": "How many points is a touchdown worth in American football?",
     "answer": "6",
     "steps": "- Touchdown is main scoring play\n- Worth 6 points\n- Extra point or 2-point conversion possible after\n- But touchdown alone is 6 points"},

    # Historical Dates
    {"question": "In what year did World War II end?",
     "answer": "1945",
     "steps": "- War in Europe ended in May 1945\n- Japan surrendered in August 1945\n- Official end was September 2, 1945\n- Therefore war ended in 1945"},
    
    {"question": "When was the Declaration of Independence signed?",
     "answer": "1776",
     "steps": "- American colonies declared independence\n- Document was approved July 4th\n- Signing began on that date\n- Year was 1776"}
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
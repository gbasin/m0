import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy

# Example data
training_data = [
    {"question": "2+2?", "answer": "4"},
    {"question": "Capital of France?", "answer": "Paris"},
    {"question": "What is 5+7?", "answer": "12"},
    {"question": "Capital of Japan?", "answer": "Tokyo"},
    {"question": "What is 3×4?", "answer": "12"},
    {"question": "Capital of Italy?", "answer": "Rome"},
    {"question": "What is 10-3?", "answer": "7"},
    {"question": "Capital of Spain?", "answer": "Madrid"}
]

def parse_answer_from_chunk(chunk):
    # If the chunk has 'Answer:', parse out the final answer
    if "Answer:" in chunk:
        return chunk.split("Answer:")[-1].strip()
    return None

def compute_reward(final_answer, ground_truth):
    return 1.0 if final_answer and final_answer.lower() == ground_truth.lower() else 0.0

def rollout_episode(model, tokenizer, question, ground_truth, max_steps=5, step_tokens=20):
    """
    Multi-step rollout:
    - Generate small CoT chunks
    - Decide whether to 'continue' or 'answer' based on if we see 'Answer:'
    - Return a list of transitions: (state_text, action_token_ids, reward).
    """
    transitions = []
    
    # Create a more structured prompt with examples
    state_text = f"""Here are some examples:

Question: What is 3+5?
Let's solve this step by step:
1. Start with 3
2. Add 5 to it
3. 3 plus 5 equals 8
Answer: 8

Question: Capital of Germany?
Let's solve this step by step:
1. Germany is a country in Europe
2. Its capital city is Berlin
Answer: Berlin

Question: {question}
Let's solve this step by step:
"""
    done = False
    
    print(f"\nStarting rollout for question: {question}")
    print(f"Ground truth answer: {ground_truth}")
    
    for step in range(max_steps):
        # Encode current state with attention mask
        inputs = tokenizer(state_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate a short chunk
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=step_tokens,
            do_sample=True,
            temperature=0.7,  # Reduced temperature for more focused sampling
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=3
        )
        
        # The newly generated tokens
        gen_ids = outputs[0][input_ids.shape[1]:]
        text_chunk = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        print(f"\nStep {step + 1}:")
        print(f"Generated chunk: {text_chunk}")
        
        # Check if chunk has an answer
        final_answer = parse_answer_from_chunk(text_chunk)
        if final_answer is not None:
            r = compute_reward(final_answer, ground_truth)
            print(f"Found answer: {final_answer}")
            print(f"Reward: {r}")
            transitions.append((state_text, gen_ids, r))
            done = True
            break
        else:
            transitions.append((state_text, gen_ids, 0.0))
            state_text += text_chunk
    
    if not done:
        print("No answer found in max steps")
        transitions[-1] = (transitions[-1][0], transitions[-1][1], 0.0)
    
    return transitions

def compute_loss_for_episode(model, tokenizer, transitions, baseline=0.0):
    total_loss = 0.0
    for state_text, action_ids, reward in transitions:
        # Encode state with attention mask
        inputs = tokenizer(state_text, return_tensors="pt", padding=True)
        state_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get the full sequence including the action
        full_ids = torch.cat([state_ids[0], action_ids], dim=0).unsqueeze(0)
        full_attention_mask = torch.ones_like(full_ids)
        
        # Forward pass with attention mask
        outputs = model(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
            labels=full_ids
        )
        
        # Get logits for action tokens only
        logits = outputs.logits[:, :-1, :].contiguous()
        target_ids = full_ids[:, 1:].contiguous()
        
        # Calculate loss only for the action portion
        action_start = state_ids.size(1) - 1
        action_logits = logits[:, action_start:, :]
        action_targets = target_ids[:, action_start:]
        
        # Compute cross entropy loss for action tokens
        nll = F.cross_entropy(
            action_logits.view(-1, action_logits.size(-1)),
            action_targets.view(-1),
            reduction="sum"
        )
        
        advantage = (reward - baseline)
        step_loss = advantage * nll
        print(f"Step loss: {step_loss.item():.3f} (reward={reward:.1f}, nll={nll.item():.3f})")
        total_loss += step_loss
    
    return total_loss

def train_simple_reinforce(model, tokenizer, data, epochs=5, lr=1e-5, baseline=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("\nStarting training...")
    print(f"Learning rate: {lr}")
    print(f"Number of examples: {len(data)}")
    print(f"Epochs: {epochs}")
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch} ===")
        total_loss_val = 0.0
        total_reward = 0.0
        for i, example in enumerate(data):
            print(f"\nExample {i+1}/{len(data)}")
            transitions = rollout_episode(model, tokenizer, example["question"], example["answer"])
            total_reward += transitions[-1][2]
            
            print("\nComputing loss...")
            loss = compute_loss_for_episode(model, tokenizer, transitions, baseline=baseline)
            print(f"Total loss for example: {loss.item():.3f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_val += loss.item()
            
        avg_loss = total_loss_val / len(data)
        avg_reward = total_reward / len(data)
        print(f"\nEpoch {epoch} Summary:")
        print(f"Average Loss: {avg_loss:.3f}")
        print(f"Average Reward: {avg_reward:.3f}")
        print("-" * 50)

def main():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Train with simple REINFORCE
    train_simple_reinforce(model, tokenizer, training_data, epochs=10, lr=1e-5, baseline=0.0)

if __name__ == "__main__":
    main()
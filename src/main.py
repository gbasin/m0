import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy

# Example data
training_data = [
    {"question": "2+2?", "answer": "4"},
    {"question": "Capital of France?", "answer": "Paris"},
    # More...
]

def parse_answer_from_chunk(chunk):
    # If the chunk has 'Answer:', parse out the final answer
    if "Answer:" in chunk:
        return chunk.split("Answer:")[-1].strip()
    return None

def compute_reward(final_answer, ground_truth):
    return 1.0 if final_answer and final_answer.lower() == ground_truth.lower() else 0.0

def rollout_episode(model, tokenizer, question, ground_truth, max_steps=5, step_tokens=10):
    """
    Multi-step rollout:
    - Generate small CoT chunks
    - Decide whether to 'continue' or 'answer' based on if we see 'Answer:'
    - Return a list of transitions: (state_text, action_token_ids, reward).
    """
    device = next(model.parameters()).device
    transitions = []
    
    state_text = f"Question: {question}\nChain-of-thought:"
    done = False
    
    for step in range(max_steps):
        # Encode current state with attention mask
        inputs = tokenizer(state_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Generate a short chunk
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=step_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # The newly generated tokens
        gen_ids = outputs[0][input_ids.shape[1]:]
        text_chunk = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        # Check if chunk has an answer
        final_answer = parse_answer_from_chunk(text_chunk)
        if final_answer is not None:
            # We have a candidate final answer -> compute reward
            r = compute_reward(final_answer, ground_truth)
            transitions.append((state_text, gen_ids, r))
            done = True
            break
        else:
            # Still generating chain-of-thought
            transitions.append((state_text, gen_ids, 0.0))
            state_text += text_chunk
    
    # If never answered, assign 0 reward for the last transition
    if not done:
        transitions[-1] = (transitions[-1][0], transitions[-1][1], 0.0)
    
    return transitions

def compute_loss_for_episode(model, tokenizer, transitions, baseline=0.0):
    """
    For each transition, compute log pi(action|state) * (reward - baseline).
    Sum up as a total REINFORCE loss (negative since we do gradient descent on -objective).
    """
    device = next(model.parameters()).device
    total_loss = 0.0
    for state_text, action_ids, reward in transitions:
        # Encode state
        state_ids = tokenizer.encode(state_text, return_tensors="pt").to(device)
        # We want the logits for the next tokens
        with torch.no_grad():
            state_outputs = model(state_ids, labels=None)
        
        # Now feed the state+action to get log probs for the action
        # The simplest way is to get the log probs for each token in the action
        # offset the hidden states or do a direct forward for the entire sequence
        full_ids = torch.cat([state_ids[0], action_ids], dim=0).unsqueeze(0)
        outputs = model(full_ids, labels=full_ids)
        # outputs.logits shape: [batch_size, seq_len, vocab_size]
        # Shift them so we get token-level log probs
        logits = outputs.logits[:, :-1, :].contiguous()  # all but last
        target_ids = full_ids[:, 1:].contiguous()        # all but first
        # Negative log-likelihood for the action tokens
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="none")
        # Sum over the action portion only (the new tokens)
        # The new tokens = length of action_ids
        # We'll take the last len(action_ids) tokens from nll
        action_nll = nll[-len(action_ids):].sum()
        
        advantage = (reward - baseline)
        step_loss = advantage * action_nll
        total_loss += step_loss
    
    return total_loss

def train_simple_reinforce(model, tokenizer, data, epochs=5, lr=1e-5, baseline=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    
    for epoch in range(epochs):
        total_loss_val = 0.0
        total_reward = 0.0
        for example in data:
            transitions = rollout_episode(model, tokenizer, example["question"], example["answer"])
            # Compute total reward in the final transition (the model either answered or didn't)
            total_reward += transitions[-1][2]
            loss = compute_loss_for_episode(model, tokenizer, transitions, baseline=baseline)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_val += loss.item()
        avg_loss = total_loss_val / len(data)
        avg_reward = total_reward / len(data)
        print(f"Epoch {epoch}: Avg Loss={avg_loss:.3f}  Avg Reward={avg_reward:.3f}")

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
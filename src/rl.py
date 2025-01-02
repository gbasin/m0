import torch
import torch.nn.functional as F

def parse_answer_from_chunk(chunk, previous_text=""):
    """Look for answer between <ans> tags, handling split tags across chunks"""
    # Combine with previous text to handle split tags
    full_text = previous_text + chunk
    
    # Check if we have a complete answer
    if "<ans>" in full_text and "</ans>" in full_text:
        answer = full_text.split("<ans>")[-1].split("</ans>")[0].strip()
        return answer, ""  # Reset buffer since we found complete answer
    
    # If we have an opening tag but no closing tag, keep the partial answer
    if "<ans>" in full_text and "</ans>" not in full_text:
        return None, full_text
    
    # If we have neither tag or only closing tag, just keep last few characters
    # in case we're in the middle of a tag
    return None, full_text[-10:] if len(full_text) > 10 else full_text

def compute_reward(final_answer, ground_truth):
    """Compute reward for an answer"""
    if final_answer is None:
        return 0.0
    
    print(f"Comparing answer: '{final_answer}' with ground truth: '{ground_truth}'")
    return 1.0 if final_answer.lower() == ground_truth.lower() else 0.0

def rollout_episode(model, tokenizer, question, ground_truth, max_steps=3, step_tokens=20):
    """Multi-step rollout with a small model"""
    transitions = []
    
    # Format that better matches Pythia's training
    state_text = f"""Below are examples of solving problems step by step. Each solution includes reasoning and an answer between <ans> tags.

Input: What is 3+5?
Output:
- Start with 3
- Add 5
- 3 + 5 = 8
Therefore, <ans>8</ans>

Input: What is the capital of Germany?
Output:
- Germany is a country in Europe
- Its capital city is Berlin
Therefore, <ans>Berlin</ans>

Input: {question}
Output:
"""
    done = False
    answer_buffer = ""  # Keep track of partial answers across chunks
    
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
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.2
        )
        
        # The newly generated tokens
        gen_ids = outputs[0][input_ids.shape[1]:]
        text_chunk = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        print(f"\nStep {step + 1}:")
        print(f"Generated chunk: {text_chunk}")
        
        # Check for answer, passing in previous buffer
        final_answer, new_buffer = parse_answer_from_chunk(text_chunk, answer_buffer)
        answer_buffer = new_buffer
        
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
        # Check if we have a partial answer in the buffer
        final_answer, _ = parse_answer_from_chunk(answer_buffer)
        if final_answer is not None:
            r = compute_reward(final_answer, ground_truth)
            transitions[-1] = (transitions[-1][0], transitions[-1][1], r)
        else:
            transitions[-1] = (transitions[-1][0], transitions[-1][1], -0.5)
    
    return transitions

def compute_loss_for_episode(model, tokenizer, transitions, baseline=0.0):
    """Compute loss for an episode using REINFORCE"""
    total_loss = torch.tensor(0.0)  # Initialize as tensor
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
            reduction="mean"  # Changed from sum to mean
        )
        
        advantage = torch.tensor(reward - baseline)
        step_loss = -advantage * nll  # Negative since we want to maximize reward
        print(f"Step loss: {step_loss.item():.3f} (reward={reward:.1f}, nll={nll.item():.3f})")
        total_loss += step_loss
    
    return total_loss

def train_reinforce(model, tokenizer, data, epochs=5, lr=1e-5, baseline=0.0):
    """Train the model using REINFORCE"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("\nStarting REINFORCE training...")
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
    
    return model 
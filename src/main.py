import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import create_training_data
from sft import train_supervised
from rl import train_reinforce

def main():
    model_name = "EleutherAI/pythia-410m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # First do supervised fine-tuning
    print("\nStarting supervised training...")
    formatted_data = create_training_data()
    model = train_supervised(model, tokenizer, formatted_data, epochs=3, batch_size=4, learning_rate=1e-5)
    
    # Then optionally do RL fine-tuning
    # print("\nStarting RL training...")
    # model = train_reinforce(model, tokenizer, training_data, epochs=5, lr=1e-5, baseline=0.0)

if __name__ == "__main__":
    main()
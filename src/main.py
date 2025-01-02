import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import create_training_data
from sft import train_supervised
from rl import train_reinforce
import os
import glob

def get_latest_checkpoint():
    """Find the most recent checkpoint directory"""
    results_dir = "./results"
    if not os.path.exists(results_dir):
        return None
    
    # Find all SFT directories
    sft_dirs = glob.glob(os.path.join(results_dir, "sft_*"))
    if not sft_dirs:
        return None
    
    # Get the most recent directory
    latest_dir = max(sft_dirs, key=os.path.getctime)
    
    # Check for final model
    final_dir = os.path.join(latest_dir, "final")
    if os.path.exists(final_dir):
        return final_dir
    
    # Otherwise check for checkpoints
    checkpoint_dirs = glob.glob(os.path.join(latest_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    
    return max(checkpoint_dirs, key=os.path.getctime)

def main():
    model_name = "EleutherAI/pythia-410m"
    
    # Check for latest checkpoint
    checkpoint_dir = get_latest_checkpoint()
    if checkpoint_dir:
        print(f"\nLoading from checkpoint: {checkpoint_dir}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    else:
        print(f"\nStarting fresh with model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # First do supervised fine-tuning
    print("\nStarting supervised training...")
    formatted_data = create_training_data()
    model, save_path = train_supervised(model, tokenizer, formatted_data, epochs=10, batch_size=2, learning_rate=5e-5)
    print(f"\nTraining complete! Model saved to: {save_path}")
    
    # Then optionally do RL fine-tuning
    # print("\nStarting RL training...")
    # model = train_reinforce(model, tokenizer, training_data, epochs=5, lr=1e-5, baseline=0.0)

if __name__ == "__main__":
    main()
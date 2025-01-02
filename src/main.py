import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import create_training_data, training_data
from sft import train_supervised
from rl import train_reinforce
import os
import glob
from datetime import datetime
import argparse

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train language model with SFT and/or RL')
    parser.add_argument('--sft', action='store_true', help='Run supervised fine-tuning')
    parser.add_argument('--rl', action='store_true', help='Run reinforcement learning')
    parser.add_argument('--model', default="EleutherAI/pythia-410m", help='Base model to use')
    parser.add_argument('--sft-epochs', type=int, default=10, help='Number of SFT epochs')
    parser.add_argument('--rl-epochs', type=int, default=3, help='Number of RL epochs')
    parser.add_argument('--sft-lr', type=float, default=5e-5, help='SFT learning rate')
    parser.add_argument('--rl-lr', type=float, default=1e-6, help='RL learning rate')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training')
    args = parser.parse_args()
    
    # If neither specified, run both
    if not args.sft and not args.rl:
        args.sft = True
        args.rl = True
    
    # Load model
    checkpoint_dir = get_latest_checkpoint()
    if checkpoint_dir:
        print(f"\nLoading from checkpoint: {checkpoint_dir}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, use_cache=False)
    else:
        print(f"\nStarting fresh with model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, use_cache=False)
    
    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Run supervised fine-tuning if requested
    if args.sft:
        print("\nStarting supervised training...")
        formatted_data = create_training_data()
        model, save_path = train_supervised(
            model, 
            tokenizer, 
            formatted_data, 
            epochs=args.sft_epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.sft_lr
        )
        print(f"\nSFT complete! Model saved to: {save_path}")
    
    # Run RL fine-tuning if requested
    if args.rl:
        print("\nStarting RL training...")
        model = train_reinforce(
            model, 
            tokenizer, 
            training_data, 
            epochs=args.rl_epochs, 
            lr=args.rl_lr, 
            baseline=0.5
        )
        
        # Save final model after RL
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = f"./results/rl_{timestamp}/final"
        os.makedirs(final_dir, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"\nFinal model after RL saved to: {final_dir}")

if __name__ == "__main__":
    main()
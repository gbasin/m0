import torch
from transformers import TrainingArguments, Trainer
import os
from datetime import datetime
import numpy as np
from typing import Dict, Any

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute metrics for logging"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Compute loss (we care about perplexity and loss)
    loss = np.mean(np.where(labels != -100, 
                           np.equal(predictions, labels).astype(np.float32),
                           0.0))
    perplexity = np.exp(-loss)
    
    return {
        "loss": loss,
        "perplexity": perplexity,
    }

def train_supervised(model, tokenizer, formatted_data, epochs=3, batch_size=4, learning_rate=1e-5):
    """Train the model using supervised learning"""
    
    # Create a simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer):
            # Ensure consistent padding across all examples
            self.encodings = tokenizer(
                texts, 
                truncation=True, 
                padding='max_length',  # Changed from True to 'max_length'
                max_length=512, 
                return_tensors="pt"
            )
        
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item
        
        def __len__(self):
            return len(self.encodings.input_ids)
    
    # Create dataset
    full_dataset = SimpleDataset(formatted_data, tokenizer)
    
    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/sft_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=1,  # Log every step
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=50,  # Reduced frequency - evaluate every 50 steps
        save_steps=50,  # Save at same frequency as eval
        save_total_limit=2,  # Keep only last 2 checkpoints
        report_to=["tensorboard"],
        logging_first_step=True,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        do_eval=True,
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
        label_names=["labels"],
        max_grad_norm=1.0,
        warmup_steps=10,  # Reduced warmup steps given small dataset
        save_safetensors=True,
        resume_from_checkpoint=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # Add metrics computation
    )
    
    # Train the model
    print(f"\nStarting training...")
    print(f"Output directory: {output_dir}")
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Tensorboard logs will be in: {training_args.logging_dir}")
    
    trainer.train()
    
    # Save final model and tokenizer
    final_output_dir = f"{output_dir}/final"
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"\nSaved final model to {final_output_dir}")
    
    return model, final_output_dir 
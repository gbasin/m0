import torch
from transformers import TrainingArguments, Trainer

def train_supervised(model, tokenizer, formatted_data, epochs=3, batch_size=4, learning_rate=1e-5):
    """Train the model using supervised learning"""
    
    # Create a simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item
        
        def __len__(self):
            return len(self.encodings.input_ids)
    
    # Create dataset
    dataset = SimpleDataset(formatted_data, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1,
        save_total_limit=2,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train the model
    trainer.train()
    
    return model 
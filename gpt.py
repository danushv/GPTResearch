from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

def main():
    # Load the dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Ensure the tokenizer will return the tensors in PyTorch format and add a pad token if it's not already there
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize the texts
        tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        # In language modeling, the labels are the same as the input_ids
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # Train the model
    trainer.train()

    # Example of generating text after fine-tuning
    prompt = "The ancient library"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    # Generate text using the fine-tuned model
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()

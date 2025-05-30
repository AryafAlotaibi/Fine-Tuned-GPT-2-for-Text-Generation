
!pip install gradio transformers

"""# **Dataset Processing and Model Implementation**"""

import os
import pandas as pd
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
import math

# 1. Load data
df = pd.read_csv("merged_dataset.csv")
all_text = ""

# Combine data into a single text
for col in df.select_dtypes(include="object").columns:
    all_text += "\n".join(df[col].dropna().astype(str)) + "\n"

# 2. Split data into training and validation sets
train_text, eval_text = train_test_split(all_text.split("\n"), test_size=0.1, random_state=42)

# Save combined data into text files
with open("train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_text))
with open("eval.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(eval_text))

# 3. Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # padding
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 4. Load and organize data into training format
def load_dataset(tokenizer, file_path, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

train_dataset = load_dataset(tokenizer, "train.txt")
eval_dataset = load_dataset(tokenizer, "eval.txt")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Configure training parameters
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    save_steps=50,
    save_total_limit=1,
    logging_steps=10,
    logging_first_step=True,
    report_to="none"
)

# 6. Define perplexity calculation function
def compute_perplexity(eval_loss):
    return math.exp(eval_loss)

# 7. Create Trainer
class CustomTrainer(Trainer):
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        eval_loss = logits.mean()
        perplexity = compute_perplexity(eval_loss)
        return {"eval_loss": eval_loss, "perplexity": perplexity}

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# 8. Start training
trainer.train()

# 9. Save trained model
model_path1= "./gpt2-finetuned"
trainer.save_model(model_path1)
tokenizer.save_pretrained(model_path1)

# 10. Evaluation and metric calculation
eval_results = trainer.evaluate()
print("\n📊 Evaluation Results:")
print(f"Loss: {eval_results['eval_loss']:.4f}")
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

"""# **Example 1**"""

from transformers import pipeline

generator = pipeline("text-generation", model=model_path1, tokenizer=model_path1)

prompt1 = "My friend"


outputs1 = generator(
    prompt1,
    max_length=30,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    truncation=True
)


print(outputs1[0]["generated_text"].replace("\n", " "))

"""# **Example 2**"""

from transformers import pipeline

generator = pipeline("text-generation", model=model_path1, tokenizer=model_path1)

prompt2 = "My book"

outputs2 = generator(
    prompt2,
    max_length=30,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    truncation=True
)
print(outputs2[0]["generated_text"].replace("\n", " "))

"""# **API  Implementation**"""

from transformers import pipeline
import gradio as gr

# Load the model and tokenizer
model =model_path1
generator = pipeline("text-generation", model=model_path1, tokenizer=model_path1)

# Define the text generation function
def generate_text(prompt):
    outputs = generator(
        prompt,
        max_length=20,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        truncation=True
    )
    return outputs[0]["generated_text"].replace("\n", " ")

# Create Gradio interface
interface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Textbox(label="Generated Text"),
    title="Text Generation API",
    description="Enter a prompt to generate text using a local Hugging Face model.",
    allow_flagging="never",
    live=False
)

interface.launch()
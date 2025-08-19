from torch.utils.tensorboard import SummaryWriter
from datasets import load_metric
# =====================
# LoRA Finetune Config
# =====================
MODEL_NAME = "ByteDance-Seed/Seed-X-RM-7B"
DATA_PATH = "../Data/"  # Update as needed
OUTPUT_DIR = "./lora_output"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512
SAVE_STEPS = 100
LOGGING_STEPS = 10
USE_FP16 = True

# =====================
# LoRA Finetune Script
# =====================
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os

import pandas as pd

def load_dataset(data_path):
	# Expects CSV files with 'Japanese' and 'Vietnamese' columns
	all_data = []
	for fname in os.listdir(data_path):
		if fname.endswith('.csv'):
			df = pd.read_csv(os.path.join(data_path, fname))
			for _, row in df.iterrows():
				# Format as instruction tuning: input is Japanese, output is Vietnamese
				prompt = f"Translate the following Japanese sentence to Vietnamese: {row['Japanese']}\nAnswer: {row['Vietnamese']}"
				all_data.append({"text": prompt})
	return all_data



def main():
	# Optionally, split your data for validation
	# Here, we use 90% for training, 10% for validation
	from sklearn.model_selection import train_test_split
	train_data, val_data = train_test_split(tokenized_data, test_size=0.1, random_state=42)

	train_dataset = SimpleDataset(train_data)
	val_dataset = SimpleDataset(val_data)
	data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

	writer = SummaryWriter(os.path.join(OUTPUT_DIR, "runs"))
	bleu = load_metric("sacrebleu")
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_NAME,
		torch_dtype=torch.float16 if USE_FP16 and device == "cuda" else torch.float32,
		device_map="auto" if device == "cuda" else None
	)

	# Prepare model for LoRA
	model = prepare_model_for_kbit_training(model)
	lora_config = LoraConfig(
		r=LORA_R,
		lora_alpha=LORA_ALPHA,
		target_modules=["q_proj", "v_proj"],  # Update as needed for model arch
		lora_dropout=LORA_DROPOUT,
		bias="none",
		task_type="CAUSAL_LM"
	)
	model = get_peft_model(model, lora_config)

	# Load and tokenize dataset
	raw_data = load_dataset(DATA_PATH)
	def tokenize_fn(example):
		return tokenizer(
			example["text"],
			truncation=True,
			max_length=MAX_SEQ_LENGTH,
			padding="max_length",
		)
	tokenized_data = list(map(tokenize_fn, raw_data))

	class SimpleDataset(torch.utils.data.Dataset):
		def __init__(self, data):
			self.data = data
		def __len__(self):
			return len(self.data)
		def __getitem__(self, idx):
			# Ensure tensors are on the correct device
			return {k: torch.tensor(v).to(device) for k, v in self.data[idx].items()}



	training_args = TrainingArguments(
		output_dir=OUTPUT_DIR,
		per_device_train_batch_size=BATCH_SIZE,
		gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
		num_train_epochs=NUM_EPOCHS,
		learning_rate=LEARNING_RATE,
		fp16=USE_FP16 and device == "cuda",
		save_steps=SAVE_STEPS,
		logging_steps=LOGGING_STEPS,
		report_to=["tensorboard"],
		logging_dir=os.path.join(OUTPUT_DIR, "runs"),
		save_total_limit=2,
		remove_unused_columns=False,
		dataloader_pin_memory=True,
	)


	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		data_collator=data_collator,
	)

	for epoch in range(1, NUM_EPOCHS + 1):
		trainer.train(resume_from_checkpoint=None)
		if epoch % 5 == 0 or epoch == NUM_EPOCHS:
			# Evaluate BLEU on validation set
			model.eval()
			preds = []
			refs = []
			for item in val_data:
				input_ids = torch.tensor(item["input_ids"]).unsqueeze(0).to(device)
				with torch.no_grad():
					output = model.generate(input_ids, max_length=MAX_SEQ_LENGTH)
				pred = tokenizer.decode(output[0], skip_special_tokens=True)
				# Extract reference from prompt
				ref = tokenizer.decode(item["labels"], skip_special_tokens=True) if "labels" in item else ""
				preds.append(pred)
				refs.append([ref])
			bleu_score = bleu.compute(predictions=preds, references=refs)["score"]
			print(f"Epoch {epoch}: BLEU = {bleu_score}")
			writer.add_scalar("BLEU/val", bleu_score, epoch)

	model.save_pretrained(OUTPUT_DIR)
	tokenizer.save_pretrained(OUTPUT_DIR)
	writer.close()

if __name__ == "__main__":
	main()

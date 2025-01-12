from data_helper import get_dataset_preprocessed
from datasets import DatasetDict, concatenate_datasets
from transformers import (set_seed, AutoTokenizer, AutoConfig, 
        AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq)
import evaluate
import numpy as np
import json
import os
import argparse

# Initialize
parser = argparse.ArgumentParser()
parser.add_argument("--train", default=False, action="store_true")
parser.add_argument("--model", default=None)
user_args = parser.parse_args()
training_mode = user_args.train
random_seed = 114514
set_seed(random_seed)
train_batch = 32
eval_batch = 64
num_epochs = 20
max_length = 128
training_tag = "test"
model_tag = "test"
tokenizer_checkpoint = "Helsinki-NLP/opus-mt-ja-en"
model_checkpoint = user_args.model
num_beams = 6
do_sample = False
evaluation_output = "results.json"
os.environ["WANDB_DISABLED"] = "true"

# Build tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

# Prepare data and preprocess
if training_mode:
    task_1_data = get_dataset_preprocessed("task_1", tokenizer, random_seed)
    opus_data = get_dataset_preprocessed("opus", tokenizer, random_seed)
    dataset_merged = DatasetDict()
    dataset_merged["train"] = concatenate_datasets([task_1_data["train"], opus_data["train"]])
    dataset_merged["test"] = concatenate_datasets([task_1_data["test"], opus_data["test"]])
    dataset_merged = dataset_merged.shuffle(seed=random_seed)
else:
    testsets = {"course":get_dataset_preprocessed("task_1", tokenizer, random_seed, True),
                "opus":get_dataset_preprocessed("opus", tokenizer, random_seed, True),
                "tatoeba":get_dataset_preprocessed("tatoeba", tokenizer, random_seed, True)}

# Build model
if model_checkpoint == None:
    config = AutoConfig.from_pretrained(tokenizer_checkpoint)
    model = AutoModelForSeq2SeqLM.from_config(config)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.generation_config.do_sample = do_sample
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Build metrics
metrics = evaluate.load("sacrebleu")
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
    result = metrics.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    reference_lens = [np.count_nonzero(label != tokenizer.pad_token_id) for label in labels]
    result["gen_len"] = np.mean(prediction_lens)
    result["ref_len"] = np.mean(reference_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Train or evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
training_args = Seq2SeqTrainingArguments(
    output_dir=f"checkpoints/{training_tag}",
    eval_strategy="steps",
    eval_steps=30000,
    save_strategy="steps",
    save_steps=30000,
    learning_rate=2e-5,
    per_device_train_batch_size=train_batch,
    per_device_eval_batch_size=eval_batch,
    weight_decay=0.01,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    generation_num_beams=num_beams,
    fp16=True,
    push_to_hub=False,
    logging_steps=1,
    generation_max_length=max_length,
)
if training_mode:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_merged["train"],
        eval_dataset=dataset_merged["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=False)
else:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=testsets["course"],
        eval_dataset=testsets["course"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    results = {}
    output_file = evaluation_output
    for name, data in testsets.items():
        trainer.eval_dataset = data
        results[name] = trainer.evaluate()
    print(results)
    with open(output_file, "r", encoding="utf-8") as file:
        saved = json.load(file)
    saved[model_tag] = results
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(saved, file, indent=4, ensure_ascii=False)
from data_helper import get_dataset_preprocessed
from datasets import DatasetDict, concatenate_datasets
from transformers import (set_seed, AutoTokenizer, AutoConfig, 
        AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq)
import evaluate
import numpy as np
import json
import os

# Initialize
training_mode = False
random_seed = 114514
set_seed(random_seed)
train_batch = 32
eval_batch = 64
num_epochs = 20
max_length = 128
training_tag = "test"
model_tag = "raw_opus_merge"
tokenizer_checkpoint = "Helsinki-NLP/opus-mt-ja-en"
model_checkpoint = f"checkpoints/{model_tag}/checkpoint-540000" 
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
# # Dataset task_1
# data = []
# with open('baseline/data/raw/eng_jpn.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         japanese, english = line.strip().split('\t')
#         data.append({"jpn": japanese, "eng": english})
# df = pd.DataFrame(data)
# dataset = Dataset.from_pandas(df)
# dataset = dataset.train_test_split(test_size=0.2, seed=random_seed)
# def preprocess_function(examples):
#     inputs = [example for example in examples["jpn"]]
#     targets = [example for example in examples["eng"]]
#     model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
#     return model_inputs
# dataset_preprocessed = dataset.map(preprocess_function, batched=True)

# # Dataset Opus
# opus_dataset = load_dataset("opus100", "en-ja")
# # opus_dataset["train"] = opus_dataset["train"].train_test_split(test_size=data_cut, seed=random_seed)["test"]
# def preprocess_opus(examples):
#     inputs = [example["ja"] for example in examples["translation"]]
#     targets = [example["en"] for example in examples["translation"]]
#     model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
#     return model_inputs
# opus_preprocessed = opus_dataset.map(preprocess_opus, batched=True)

# # Prepare tatoeba for test
# opus_test = load_dataset("opus100","en-ja",split="test")
# opus_test = opus_test.map(preprocess_opus, batched=True)
# tatoeba = load_dataset("tatoeba", lang1="en", lang2="ja", trust_remote_code=True)
# tatoeba_test = tatoeba["train"].train_test_split(test_size=0.05)["test"]
# tatoeba_test = tatoeba_test.map(preprocess_opus, batched=True)
# testsets = {"course":dataset_preprocessed["test"],
#             "opus":opus_test,"tatoeba":tatoeba_test}

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


# 优化性能：1.增加数据 2.调整模型大小 3.decoding 4.针对task/data 5.加入单个词
# 优化实验：1.metrics 2.evaluate加速(batch?)　3.汉字和假名的区别
# 对比实验：...
# 1.数据：在原始数据上收敛 在100w的opus上收敛？ 只用opus一部分
# 2.数据的组合
# 3.Decoding
# 4.Demo
# 5.成体系的检验
from datasets import Dataset, load_dataset
import pandas as pd

def get_dataset_preprocessed(name, tokenizer, seed, test_only=False):
    max_length = 128
    if name == "task_1":
        data = []
        with open('data/eng_jpn.txt', 'r', encoding='utf-8') as file:
            for line in file:
                japanese, english = line.strip().split('\t')
                data.append({"jpn": japanese, "eng": english})
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        def preprocess_function(examples):
            inputs = [example for example in examples["jpn"]]
            targets = [example for example in examples["eng"]]
            model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
            return model_inputs
        dataset_preprocessed = dataset.map(preprocess_function, batched=True)
        if test_only:
            return dataset_preprocessed["test"]
        return dataset_preprocessed
    def preprocess_opus(examples):
        inputs = [example["ja"] for example in examples["translation"]]
        targets = [example["en"] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        return model_inputs
    if name == "opus":
        if test_only:
            opus_test = load_dataset("opus100","en-ja",split="test")
            opus_test = opus_test.map(preprocess_opus, batched=True)
            return opus_test
        opus_dataset = load_dataset("opus100", "en-ja")
        opus_preprocessed = opus_dataset.map(preprocess_opus, batched=True)
        return opus_preprocessed
    if name == "tatoeba":
        if not test_only:
            raise ValueError("We only use tatoeba for evaluation!")
        tatoeba = load_dataset("tatoeba", lang1="en", lang2="ja", trust_remote_code=True)
        tatoeba_test = tatoeba["train"].train_test_split(test_size=0.05, seed=seed)["test"]
        tatoeba_test = tatoeba_test.map(preprocess_opus, batched=True)
        return tatoeba_test
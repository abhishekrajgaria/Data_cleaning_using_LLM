import json
import time
import torch
import random
import evaluate
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from constants import *
from torch.optim import Adam

start = time.perf_counter()


from transformers import T5Tokenizer, T5ForConditionalGeneration, T5TokenizerFast

from torch.utils.data import Dataset, DataLoader, RandomSampler


train_df = pd.read_json("context_query_QA_diesease_1.json")

test_df = pd.read_json("test_context_query_QA_diesease_1.json")

DEVICE = "cuda:0"


model_checkpoint_filename = "/scratch/general/vast/u1471428/dcp/models/qa_models"
tokenizer_checkpoint_filename = "/scratch/general/vast/u1471428/dcp/models/tokenizer"


MODEL = T5ForConditionalGeneration.from_pretrained(model_checkpoint_filename).to(DEVICE)
TOKENIZER = T5Tokenizer.from_pretrained(tokenizer_checkpoint_filename)

# TOKENIZER = T5TokenizerFast.from_pretrained("google/flan-t5-large")
# MODEL = T5ForConditionalGeneration.from_pretrained(
#     "google/flan-t5-large", cache_dir=custom_cache_dir, return_dict=True
# ).to(DEVICE)

OPTIMIZER = Adam(MODEL.parameters(), lr=0.0001)
Q_LEN = 2048  # Question Length
T_LEN = 256  # Target Length
BATCH_SIZE = 2


class QA_Dataset(Dataset):
    def __init__(self, tokenizer, dataframe, q_len, t_len):

        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len

        self.questions = dataframe["query"]
        self.answer = dataframe["answer"]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answer[idx]

        question_tokenized = self.tokenizer(
            question,
            max_length=self.q_len,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )
        answer_tokenized = self.tokenizer(
            answer,
            max_length=self.t_len,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )

        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100

        return {
            "input_ids": torch.tensor(
                question_tokenized["input_ids"], dtype=torch.long
            ).to(DEVICE),
            "attention_mask": torch.tensor(
                question_tokenized["attention_mask"], dtype=torch.long
            ).to(DEVICE),
            "labels": labels.to(DEVICE),
            "decoder_attention_mask": torch.tensor(
                answer_tokenized["attention_mask"], dtype=torch.long
            ).to(DEVICE),
        }


print("after func def api")
# train_data, val_data = train_test_split(data, test_size=0.25, random_state=42)
print(train_df.shape)
qca_dataset = QA_Dataset(TOKENIZER, train_df, Q_LEN, T_LEN)

train_loader = DataLoader(qca_dataset, batch_size=BATCH_SIZE)
# val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE)

train_loss = 0
val_loss = 0
train_batch_count = 0
val_batch_count = 0
start_epoch = 0

for epoch in range(4):
    MODEL.train()
    end = time.perf_counter()
    elapsed = end - start
    print(f"Time taken: {elapsed:.6f} seconds")
    for batch in tqdm(train_loader, desc="Training batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        train_loss += outputs.loss.item()
        print(f"{epoch+1}/{4} -> Train loss: {outputs.loss}")

        train_batch_count += 1
    torch.cuda.empty_cache()


model_checkpoint_filename = "/scratch/general/vast/u1471428/dcp/models/qca_models"
tokenizer_checkpoint_filename = (
    "/scratch/general/vast/u1471428/dcp/models/qca_tokenizer"
)

MODEL.save_pretrained(model_checkpoint_filename)
TOKENIZER.save_pretrained(tokenizer_checkpoint_filename)


def predict_answer(question, ref_answer=None):
    inputs = TOKENIZER(
        question,
        max_length=Q_LEN,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )

    input_ids = (
        torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0).to(DEVICE)
    )
    attention_mask = (
        torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0).to(DEVICE)
    )

    outputs = MODEL.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=100
    )

    predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)

    if ref_answer:
        # Load the Bleu metric
        bleu = evaluate.load("google_bleu")
        score = bleu.compute(predictions=[predicted_answer], references=[ref_answer])

        print("Question: \n", question)
        return {
            "Reference Answer: ": ref_answer,
            "Predicted Answer: ": predicted_answer,
            "BLEU Score: ": score,
        }
    else:
        return predicted_answer


MODEL = T5ForConditionalGeneration.from_pretrained(model_checkpoint_filename).to(DEVICE)
TOKENIZER = T5Tokenizer.from_pretrained(tokenizer_checkpoint_filename)


question = []
answers = []
correct_answers = []
predictions = []
references = []
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


for index, row in test_df.iterrows():
    qu1 = row["query"]
    a1 = row["answer"]
    query = qu1
    ans = predict_answer(query, "")
    question.append(query)
    answers.append(ans)
    correct_answers.append(a1)
    predictions.append(ans)
    references.append([a1])


precision = precision_score(references, predictions, average="weighted")
recall = recall_score(references, predictions, average="weighted")
f1 = f1_score(references, predictions, average="weighted")


print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


bleu = evaluate.load("google_bleu")
rouge = evaluate.load("rouge")

bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)

print("BLEU Score:", bleu_score)
print("ROUGE Score:", rouge_score)


data = {"Questions": question, "Response": answers, "Correct_Response": correct_answers}
qca = pd.DataFrame(data)
print("here")
# print(qa)
qca.to_csv("disease_augmented_with_context.csv")

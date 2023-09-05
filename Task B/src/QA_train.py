from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
from QQA23_TaskB_read_write_qrcd import * 
import pandas as pd
import datasets 
import torch

max_length = 256 # The maximum length of a feature (question and context)
doc_stride = 64 # The authorized overlap between two part of the context when splitting it is needed.
lr = 3e-5

#Arabic best availble pre-train model


#epoch = 1
#batch_size = 8
#model = "wissamantoun/araelectra-base-artydiqa" # run 1 : pAP@10 = 0.0.437

epoch = 30
batch_size = 4
model = "ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA" # run 3 : pAP@10 = 0.469


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ar_tokenizer = AutoTokenizer.from_pretrained(model)
ar_model = AutoModelForQuestionAnswering.from_pretrained(model).to(device)


def prepare_train_features(examples):
    global ar_tokenizer
    tokenized_examples = ar_tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True)
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(ar_tokenizer.cls_token_id)
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples    


def create_dataset(train_passage_question_objects):
    datasets_ = []
    for passage_question_object in train_passage_question_objects:
        for r in passage_question_object["answers"]:
            # print(r)
            ans = dict({'answer_start': [r["start_char"]], 'text': [r["text"]]})
            datasets_.append(
                dict(
                {"id": passage_question_object["pq_id"],
                "context": passage_question_object["passage"],
                "question":passage_question_object["question"],
                "answers": ans
                    }))

    
    datasets_ = pd.DataFrame(datasets_)
    train_dataset = datasets.Dataset.from_dict(datasets_)
    return train_dataset

def train_QA():
    print(device)
    print(model)
    train_set_file = "data/QQA23_TaskB_qrcd_v1.2_train_preprocessed.jsonl"
    dev_set_file = "data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl"

    train_passage_question_objects  = load_jsonl(train_set_file)
    dev_passage_question_objects = load_jsonl(dev_set_file)

    train_dataset = create_dataset(train_passage_question_objects)
    dev_dataset = create_dataset(dev_passage_question_objects)
    my_dataset_dict = datasets.DatasetDict({"train":train_dataset, "dev" : dev_dataset})
    tokenized_ds = my_dataset_dict.map(prepare_train_features, batched=True, remove_columns=my_dataset_dict["train"].column_names)

    args = TrainingArguments(
        f"result",
        evaluation_strategy = "epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.0001,
        save_strategy = "epoch",
        load_best_model_at_end=True)


    trainer = Trainer(
    model=ar_model,
    args=args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['dev'],
    tokenizer=ar_tokenizer)

    # start training
    trainer.train()
    
    model_path = "Quran_QA_model"
    trainer.save_model(model_path)
    
    return ar_tokenizer, ar_model
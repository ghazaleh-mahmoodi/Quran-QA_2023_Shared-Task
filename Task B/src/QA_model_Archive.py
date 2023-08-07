from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
import datasets 
import pandas as pd
from read_write_qrcd import * 
import json
import torch

max_length = 256 # The maximum length of a feature (question and context)
doc_stride = 64 # The authorized overlap between two part of the context when splitting it is needed.
batch_size = 8
lr = 3e-5
epoch = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_train_features(examples):
    tokenized_examples = ar_tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,)
    
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

#model = "aubmindlab/bert-large-arabertv02" # run 0 : pAP@10 = 0.289
#model = "wissamantoun/araelectra-base-artydiqa" # run 1 : pAP@10 = 0.437
#model = "salti/AraElectra-base-finetuned-ARCD" # run 2 : pAP@10 = 0.397
#model = "ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA" # run 3 : pAP@10 = 0.435
#model = "timpal0l/mdeberta-v3-base-squad2" # run 4 : pAP@10 = 0.367
#model = "gfdgdfgdg/arap_qa_bert" # run 5 : pAP@10 = 0.184
#model = "gfdgdfgdg/arap_qa_bert_large_v2" # run6 : pAP@10 = 0.372
#model = "gfdgdfgdg/arap_qa_bert_v2" # run7 : pAP@10 = 0.344
#model = "zohaib99k/Bert_Arabic-SQuADv2-QA" # run8 : pAP@10 = 0.435
#model = "arabi-elidrisi/ArabicDistilBERT_QA" #run 9 : pAP@10 = 0.343
#model = "MMars/Question_Answering_AraBERT_xtreme_ar" #run 10 : pAP@10 = 0.337
model = "abdalrahmanshahrour/ArabicQA" # run 11 : pAP@10 = 0.304
model = "abdalrahmanshahrour/xtremeQA-ar" # run 12 : pAP@10 = 0.120
####"LoaiThamer2/GPT-4"

ar_tokenizer = AutoTokenizer.from_pretrained(model)
ar_model = AutoModelForQuestionAnswering.from_pretrained(model).to(device)

train_set_file = "data/QQA23_TaskB_qrcd_v1.2_train_preprocessed.jsonl"
dev_set_file = "data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl"

train_passage_question_objects  = load_jsonl(train_set_file)
dev_passage_question_objects = load_jsonl(dev_set_file)


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
    weight_decay=0.0001)


trainer = Trainer(
model=ar_model,
args=args,
train_dataset=tokenized_ds['train'],
eval_dataset=tokenized_ds['dev'],
    tokenizer=ar_tokenizer)

    # start training
trainer.train()

from transformers import pipeline 
qa_pipeline = pipeline('question-answering', model=ar_model, tokenizer=ar_tokenizer, topk=10, device=device)



##submassion format
def convert_list_to_dict(listOfdict):
  final_dect ={}
  for elemnt in listOfdict:
    # final_dect[elemnt]=elemnt
    for key,value in elemnt.items():
      final_dect[key]=value
  return final_dect

def convert_char_to_token_index(passage, answers, start_char, end_char):
    tokens = passage.split()
    token_index = list(range(0, len(tokens)))
    df = pd.DataFrame({'token': tokens, 'token_index':token_index})
    answers = answers.split()

    first_token_proba = list(df[df["token"] == answers[0]]["token_index"])
    end_token_proba = list(df[df["token"] == answers[-1]]["token_index"])

    start = 0
    end = 0
    if len(first_token_proba) == 1:
        start = first_token_proba[0]
    
    if len(end_token_proba) == 1 : 
        end = end_token_proba[0]

    return start, end

def sumbmission_file(dev_passage_question_objects):

  list_dict = []
  for passage_question_objects in dev_passage_question_objects:
      pq_id = passage_question_objects['pq_id']
      question = passage_question_objects['question']
      passage = passage_question_objects['passage']
      res = qa_pipeline({'question': question, 'context': passage}, top_k=10)
      list_sorted = sorted(res, key=lambda d: d['score'],reverse=True)
      # results_attribute = [ (list_sorted[i]['answer'],list_sorted[i]['score'],i+1) for i in range(len(list_sorted))]
      results_attribute =[]
      for i in range(len(list_sorted)):
          start, end = convert_char_to_token_index(passage, list_sorted[i]['answer'], list_sorted[i]['start'],list_sorted[i]['end'])
          if start != 0  and end == 0:
              end = start + len(list_sorted[i]['answer'].split()) -1
          if start == 0 and end!=0:
              end_ = end - len(list_sorted[i]['answer'].split()) + 1
              start = start if end_<0 else end_
          if start == 0 and end == 0:
              print("pid : ", pq_id)
              end = start + len(list_sorted[i]['answer'].split()) -1
          
          results_attribute.append((list_sorted[i]['answer'], list_sorted[i]['score'], i+1, start, end))

      final_dect= [{"answer":i[0],"score":i[1],"rank":i[2], "strt_token_indx":i[3], "end_token_indx": i[4]} for i in results_attribute]
      list_dict.append({pq_id:final_dect})
    
  output_result = convert_list_to_dict(list_dict)
  return output_result


output_result = sumbmission_file(dev_passage_question_objects)
print(":HERE AND NOW")                     
output_file = 'result/Gym_run12.json'
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(output_result, outfile, indent=2, ensure_ascii=False)
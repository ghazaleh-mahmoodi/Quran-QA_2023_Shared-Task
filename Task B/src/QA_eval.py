from transformers import pipeline 
from QQA23_TaskB_read_write_qrcd import * 
import pandas as pd
import torch
import json

##submassion format
def convert_list_to_dict(listOfdict):
  final_dect ={}
  for elemnt in listOfdict:
    # final_dect[elemnt]=elemnt
    for key,value in elemnt.items():
      final_dect[key]=value
  return final_dect

def convert_char_to_token_index(passage, answers):
    '''
    It can change to better version
    '''
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

    if start != 0  and end == 0:
              end = start + len(answers) -1
    
    if start == 0 and end!=0:
        end_ = end - len(answers) + 1
        start = start if end_<0 else end_
    
    if start == 0 and end == 0:
        end = start + len(answers) -1
    return start, end


def sumbmission_file(qa_pipeline, dev_passage_question_objects):
  stopWords={'??','???','???','??','???','??','???'}
  list_dict = []
  for passage_question_objects in dev_passage_question_objects:
      
      pq_id = passage_question_objects['pq_id']
      question = passage_question_objects['question']
      passage = passage_question_objects['passage']
      res = qa_pipeline({'question': question, 'context': passage}, top_k=10)
      list_sorted = sorted(res, key=lambda d: d['score'],reverse=True)
      


      results_attribute = []
      seq_start_end = []
      for i in range(len(list_sorted)):
            if list_sorted[i]['answer'] in stopWords:
                continue
            # find_sim = False
            start, end = convert_char_to_token_index(passage, list_sorted[i]['answer'])
            # if i > 0:
            #     for t in seq_start_end:
            #         if t == (start, end):
            #             print(pq_id, t)
            #             find_sim = True
            #             continue 

            
            # seq_start_end.append((start, end))
            # if not find_sim:
            results_attribute.append((list_sorted[i]['answer'], list_sorted[i]['score'], i+1, start, end))

      #results_attribute = remove_duplicated(results_attribute)
      final_dect= [{"answer":i[0],"score":i[1],"rank":i[2], "strt_token_indx":i[3], "end_token_indx": i[4]} for i in results_attribute]
      list_dict.append({pq_id:final_dect})
    
  output_result = convert_list_to_dict(list_dict)
  return output_result


def eval_qa(ar_model, ar_tokenizer):
    dev_set_file = "data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl"
    dev_passage_question_objects = load_jsonl(dev_set_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    qa_pipeline = pipeline('question-answering', model=ar_model, tokenizer=ar_tokenizer, top_k=10, device=device)

    output_result = sumbmission_file(qa_pipeline, dev_passage_question_objects)
                    
    output_file = 'result/Gym_run12.json'
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(output_result, outfile, indent=2, ensure_ascii=False)
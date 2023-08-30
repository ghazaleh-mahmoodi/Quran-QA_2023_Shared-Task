from QA_train import *
from QA_eval import *

ar_tokenizer, ar_model = train_QA()

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#ar_tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/Gh_Mahmoudi/TaskB/result")
#ar_model = AutoModelForQuestionAnswering.from_pretrained("/home/ubuntu/Gh_Mahmoudi/TaskB/result").to(device)

dev_set_file = "data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl"
eval_qa(ar_model, ar_tokenizer, dev_set_file, output_file='result/Gym_dev.json')

test_set_file = "data/QQA23_TaskB_qrcd_v1.2_test_preprocessed.jsonl"
eval_qa(ar_model, ar_tokenizer, test_set_file, output_file='result/Gym_test1.json')

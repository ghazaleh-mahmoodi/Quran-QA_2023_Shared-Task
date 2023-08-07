from QA_train import *
from QA_eval import *

#ar_tokenizer, ar_model = train_QA()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ar_tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/Gh_Mahmoudi/TaskB/result/checkpoint-500")
ar_model = AutoModelForQuestionAnswering.from_pretrained("/home/ubuntu/Gh_Mahmoudi/TaskB/result/checkpoint-500").to(device)
eval_qa(ar_model, ar_tokenizer)
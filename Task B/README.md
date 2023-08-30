# Task B: Reading Comprehension (RC)
The task is defined as follows: Given a Qur'anic passage that consists of consecutive verses in a specific Surah of the Holy Qur'an, and a free-text question posed in MSA over that passage, a system is required to extract all answers to that question that are stated in the given passage (rather than any answer as in Qur'an QA 2022). Each answer must be a span of text extracted from the given passage. The question can be a factoid or non-factoid question. An example is shown below.

To make the task more realistic (thus challenging), some questions may not have an answer in the given passage. In such cases, the ideal system should return no answers; otherwise, it returns a ranked list of up to 10 answer spans.

## Qur'anic Reading Comprehension Dataset (QRCD)
As the development of the test set is still in progress, we only exhibit the distribution of the question-passage pairs and their associated question-passage-triplets in the training and development splits of the QRCD_v1.2 dataset (shown below). The aim is to have a split of 70%, 10%, and 20% for the training, development and test sets, respectively. 

|**Dataset** |**# Questions**|**# Question-Passage  Pairs**| **# Question-Passage-Answer  Triplets**|
|------------|:-------------:|:---------------------------:|:--------------------------------------:|
| Training   |      174      |             992             |                   1179                 |
| Development|       25      |             163             |                    220                 |

A [*reader* script](https://gitlab.com/bigirqu/quran-qa-2023/-/blob/main/Task-B/code/read_write_qrcd.py) is provided for the dataset.
<!---
| **Dataset** | **%** |**# Questions** | **# Question-Passage  Pairs** | **# Question-Passage-Answer  Triplets** |
|-------------|:-----:|:--------------:|:-----------------------------:|:---------------------------------------:|
| Training    |  70%  |      174       |             992               |                   TBA                   |
| Development |  10%  |       25       |             163               |                   TBA                   |
| Test*       |  20%  |       50       |             TBA               |                   TBA                   |
| All         |  100% |      249       |             TBA               |                   TBA                   |

*Questions of the test dataset is under development.
-->
To simplify the structure of the dataset, each tuple contains one passage, one question and a list that may contain one or more answers to that question, as shown in [this figure](https://gitlab.com/bigirqu/quran-qa-2023/-/blob/main/Task-B/data/README.md). 

Each Qur’anic passage in QRCD may have more than one occurrence; and each *passage occurrence* is paired with a different question. Likewise, each question in *QRCD* may have more than one occurrence; and each *question occurrence* is paired with a different Qur’anic passage.

The source of the Qur'anic text in QRCD is the [Tanzil project download page](https://tanzil.net/download/), which provides verified versions of the Holy Qur'an in several scripting styles. We have chosen the *simple-clean* text style of Tanzil version 1.0.2. 

## Install the required packages
```bash
pip install -r requirements.txt
```
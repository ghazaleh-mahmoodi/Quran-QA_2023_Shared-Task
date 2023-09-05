# Qur'an QA 2023 Shared Task!

This repository contains the datasets, format checkers and scorers for [Qur&#39;an QA 2023 shared task](https://sites.google.com/view/quran-qa-2023), which has two tasks.

- [Task A: Passage Retrieval (PR) task](https://gitlab.com/bigirqu/quran-qa-2023/-/tree/main/Task-A) -- given a question, find all Qur'anic passages that have potential answers.
- [Task B: Reading Comprehension (RC) task](https://gitlab.com/bigirqu/quran-qa-2023/-/tree/main/Task-B) -- given a question and a qur'anic passage, find all answers to the question.

## [Licensing and Terms of Use](https://gitlab.com/bigirqu/quran-qa-2023/-/blob/main/LICENSE)

The QRCD (Qur'anic Reading Comprehension Dataset) is distributed under the CC BY-NC-ND 4.0 License https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode

For a human-readable summary of (and not a substitute for) the above CC BY-NC-ND 4.0 License, please refer to https://creativecommons.org/licenses/by-nc-nd/4.0/

### Terms & Conditions:

- It is strictly prohibited to make any changes on the QRCD dataset, given that the answers to the questions were carefully extracted from the Holy Qur'an and then annotated by Qur'an Scholars.
- Considering that there are different schools of thought in Islam, the QRCD dataset represents a sample and not all schools of thought.
- Any suggestions for adding or refining the answers of existing questions (or adding new questions and answers) can be directed to the Qur'an QA 2023 organizers, who in turn must solicit the feedback of Qur'an scholars before effecting any updates on the QRCD dataset.
- We note that answer spans can only be extracted from their corresponding verse-based *direct* answers in the *AyaTEC* dataset. Only Qur'an scholars can decide if a verse-based answer represents a *direct* or *indirect* answer to a given question. For a formal definition of a *direct* and *indirect* answer, refer to the [*AyaTEC* paper](https://dl.acm.org/doi/abs/10.1145/3400396) (p 11).

## How to cite

* Malhas, R. and Elsayed, T., 2022. [Arabic Machine Reading Comprehension on the Holy Qur’an using CL-AraBERT](https://www.sciencedirect.com/science/article/pii/S0306457322001704). *Information Processing & Management*, 59(6), p.103068.
* Malhas, R., Mansour, W. and Elsayed, T., 2022. [Overview of the first shared task on question answering over the holy
  qur’an](https://aclanthology.org/2022.osact-1.9/). *Proceedinsg of the 5th Workshop on Open-Source Arabic Corpora and Processing Tools with Shared Tasks on Qur'an QA and Fine-Grained Hate Speech Detection*, pp. 79-87.
* Malhas, R. and Elsayed, T., 2020. [*AyaTEC*: Building a Reusable Verse-based Test Collection for Arabic Question Answering on the holy qur’an](https://www.sciencedirect.com/science/article/pii/S0306457322001704). *ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)*, 19(6), pp.1-21.

If you use the datasets of Task-A, please cite the following references

* Malhas, R., 2023. *Arabic Question Answering on the Holy Qur'an* (Doctoral dissertation).
* Malhas, R. and Elsayed, T., 2020. [*AyaTEC*: Building a Reusable Verse-based Test Collection for Arabic Question Answering on the holy qur’an](https://www.sciencedirect.com/science/article/pii/S0306457322001704). *ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)*, 19(6), pp.1-21.
* Swar, M. N.,2007.  Mushaf Al-Tafseel Al-Mawdoo’ee. Damascus: Dar Al-Fajr Al-Islami.


<!---This repository contains the following:
* The [*QRCD* (Qur'anic Reading Comprehension Dataset)](https://gitlab.com/bigirqu/quranqa/-/tree/main/datasets)
* A [*reader* script](https://gitlab.com/bigirqu/quranqa/-/tree/main/code) for the dataset.
* A [*submission checker* script]( https://gitlab.com/bigirqu/quranqa/-/tree/main/code) for checking the correctness of run files to be submitted. 
* An [*evaluation* (or *scorer*) script]( https://gitlab.com/bigirqu/quranqa/-/tree/main/code).

QRCD is composed of 1,093 tuples of question-passage pairs that are coupled with their extracted answers to constitute 1,337 question-passage-answer triplets. The distribution of the dataset into training, development and test sets is shown below.


| **Dataset** | **%** | **# Question-Passage  Pairs** | **# Question-Passage-Answer  Triplets** |
|-------------|:-----:|:-----------------------------:|:---------------------------------------:|
| Training    |  65%  |              710              |                   861                   |
| Development |  10%  |              109              |                   128                   |
| Test*       |  25%  |              ~~274~~ 238              |                   ~~348~~ 300                   |
| All         |  100% |              1,093            |                  1,337                  |

**For fairness we had to remove some questions and their answers from the test dataset used in the evaluation of the shared task. As such, 238 (instead of 274) question-passage pairs with their corresponding 300 (instead of 348) question-passage-answer triplets were used to evaluate the particpating teams. Nevertheless, we publish both test dataset versions in the [datasets folder](https://gitlab.com/bigirqu/quranqa/-/tree/main/datasets).*   

To simplify the structure of the dataset, each tuple contains one passage, one question and a list that may contain one or more answers to that question, as shown in [this figure](https://gitlab.com/bigirqu/quranqa/-/blob/main/datasets/README.md). 

Each Qur’anic passage in *QRCD* may have more than one occurrence; and each *passage occurrence* is paired with a different question. Likewise, each question in *QRCD* may have more than one occurrence; and each *question occurrence* is paired with a different Qur’anic passage.

The source of the Qur'anic text in QRCD is the [Tanzil project download page](https://tanzil.net/download/), which provides verified versions of the Holy Qur'an in several scripting styles. We have chosen the *simple-clean* text style of Tanzil version 1.0.2. 

## How to cite
If you use the *QRCD* dataset in your research, please cite the following references:
* Rana Malhas and Tamer Elsayed. Arabic Machine Reading Comprehension on the Holy Qur’an using CL-AraBERT. Information Processing & Management, 59(6), p.103068, 2022.
* Rana Malhas and Tamer Elsayed. AyaTEC: Building a Reusable Verse-Based Test Collection for Arabic Question Answering on the Holy Qur’an. ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP), 19(6), pp.1-21, 2020.
-->

# Task A: Passage Retrieval (PR)

The task is defined as follows: Given a free-text question posed in MSA and a collection of Qur'anic passages that cover the Holy Qur'an, a system is required to return a ranked list of answer-bearing passages (i.e., passages that potentially enclose the answer(s) to the given question) from this collection. The question can be a factoid or non-factoid question. An example question is shown below.

To make the task more realistic (thus challenging), some questions may not have an answer in the Holy Qur'an. In such cases, the ideal system should return no answers; otherwise, it returns a ranked list of up to 10 answer-bearing Qur'anic passages.

## [Thematic Qur&#39;an Passage Collection (QPC)](https://gitlab.com/bigirqu/quran-qa-2023/-/tree/main/Task-A/data/Thematic_QPC)

This file contains 1,266 thematic Qur'anic passages that cover the whole Holy Qur'an. Thematic passage segmentation was conducted using the Thematic Holy Qur'an [1] https://surahquran.com/tafseel-quran.html. This tsv file has the following format:

    `<passage-id>` `<passage-text>`

where the passage-id has the format: *Chapter#:StartVerse#-EndVerse#*, and the passage-text (i.e., Qur’anic text) was taken from the normalized simple-clean text style (from Tanzil 1.0.2) https://tanzil.net/download/.

## [The Training and Development Datasets](https://gitlab.com/bigirqu/quran-qa-2023/-/tree/main/Task-A/data)

These datasets are jointly composed of 199 questions from the AyaTEC dataset, of which 30  are *zero-answer* questions (i.e., questions that do not have an answer in the Holy Qur'an). They are distributed as shown in the table below.

| **Dataset** | **# Questions** | **Question-Passage Pairs**** |
| ----------------- | :-------------------: | :--------------------------------: |
| Training          |          174          |                972                |
| Development       |          25          |                160                |

## Install the required packages
```bash
pip install -r requirements.txt
```

## Run 
To run the three strategies we have provided, simply execute all the cells in main_model.ipynb notebook and observe the results.

### code structure explanation
We have three main config for multi-task learning:

<details>
           <summary>AraBERT-TSDAE-Contrastive</summary>
           <p>Unsupervised Fine-Tuning Sentence Embedding with TSDAE approach. 
             Training Bi-Encoder using Quranic Question-Passage Pairs and Mr. Tydi dataset with Contrastive and multiple negatives ranking loss. The function corresponding to this approach is gym_run1() function.
           </p>
         </details>
<details>
           <summary>AraBERT-SimCSE-Contrastive</summary>
           <p>Unsupervised Fine-Tuning Sentence Embedding with SimCSE approach.
           Training Bi-Encoder using Quranic Question-Passage Pairs  Mr. Tydi dataset Contrastive and multiple negatives ranking loss. The function corresponding to this approach is gym_run0() function.</p>
         </details>
<details>
           <summary>AraBERT-SimCSE-Triplet</summary>
           <p>Unsupervised Fine-Tuning Sentence Embedding with SimCSE approach.
           Training Bi-Encoder using Quranic Question-Passage Pairs  Mr. Tydi dataset Triplet and multiple negatives ranking loss. The function corresponding to this approach is gym_run2() function.</p>
         </details>

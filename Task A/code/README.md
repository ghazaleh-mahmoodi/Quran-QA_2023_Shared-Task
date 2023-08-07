## [Submission checker script](https://gitlab.com/bigirqu/quran-qa-2023/-/blob/main/Task-A/code/QQA23_TaskA_submission_checker.py)

It is mandatory to use this script to verify your ***run file*** (prior to submission) that it matches the TREC run format shown below.
~~~
 `<question-id>` Q0 `<passage-id>` `<rank>` `<relevance-score>` `<tag>`
~~~
<!--, i.e., have the following columns: ["qid", "Q0", "docno", "rank", "score", "tag"] -->

The expected run file is in **tsv** format (tab separated). It has a list of question ids (qid) along with their respective ranked lists of retrieved passages.

Also, the name of each submitted run file should follow the below  **naming format** .

< **TeamID_RunID.tsv**>

such that:

* **TeamID** can be an alphanumeric with a length between 3 and 9 characters
* **RunID**  can be an alphanumeric with a length between 2 and 9 characters

For example, *bigIR_run01.tsv* **is a valid name.**

## [Evaluation script](https://gitlab.com/bigirqu/quran-qa-2023/-/blob/main/Task-A/code/QQA23_TaskA_eval.py)

You can use this script to evaluate your run. You need to provide a path to the run file and the qrels (gold) file. Optionally, you can provide a file path to save the evaluation results.

Here is an example of executing the evaluation script:

```plaintext
cd Task-A/code # change directory to Task-A/code folder

python QQA23_TaskA_eval.py
    --run "./data/bigIR_run01.tsv" \
    --qrels "./data/qrels/QQA23_TaskA_qrels_train.gold" \
    --output "./data/evaluation_results.xlsx"
```

## [BM25 baseline](https://gitlab.com/bigirqu/quran-qa-2023/-/blob/main/Task-A/code/BM25_baseline.ipynb)

In this notebook, we provide a baseline to compare your work against. Basically, the baseline is BM25 (a classic retrieval model) over the QPC index. The notebook also contains some helper functions to load the index, preprocess the input text, and perform the retrieval. We show the evaluation results of the baseline at the end of the notebook.

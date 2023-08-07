
import sys
import argparse
import pandas as pd
import numpy as np
import pytrec_eval
import QQA23_TaskA_submission_checker as qqa23_sc
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

METRICS = ['map', 'recip_rank']
qrels_columns = ["qid", "Q0", "docid", "relevance"]
run_columns = ["qid", "Q0", "docid", "rank", "score", "tag"]


def read_qrels_file(qrels_file):
    # split_token = '\t' if format_checker.is_tab_sparated(qrels_file) else  "\s+"
    df_qrels = pd.read_csv(qrels_file, sep='\t', names=qrels_columns)
    df_qrels["qid"] = df_qrels["qid"].astype(str)
    df_qrels["docid"] = df_qrels["docid"].astype(str)
    return df_qrels

def read_run_file(run_file):

    # since the run is definitely is space or tab separated 
    # identify the separator token based on the file separator token
    split_token = '\t' if qqa23_sc.is_tab_sparated(run_file) else  '\s+'
    df_run = pd.read_csv(run_file, sep=split_token, names=run_columns)
    df_run["qid"] = df_run["qid"].astype(str)
    df_run["docid"] = df_run["docid"].astype(str)
    return df_run


def convert_to_dict(df, column1, column2, column3):
    '''Convert a dataframe to dictionary of dictionaries to match the TREC eval format
    column1: should be the query id column
    column2: should be the docid column
    column3: can be either the relevance column (in case of qrels) or the score column (in case of a run)
    '''
    grouped_dict = df.groupby(column1).apply(lambda x: x.set_index(column2)[column3].to_dict()).to_dict()
    # sample output:
    # qrel = {
    #     'q1': {
    #         'd1': 0,
    #         'd2': 1,
    #     },
    #     'q2': {
    #         'd2': 1,
    #         'd3': 1,
    #     },
    # }
    return grouped_dict


def get_metric_list(results_dict, metric):
    ''' Extract the values from the result dictionary and put them in a list
    '''
    values_list = [inner_dict[metric] for inner_dict in results_dict.values()]
    return values_list


def evaluate_zero_answer_questions(zero_answer_question_ids, df_run):
    ''' Evaluate the performance for the no-answer questions
    Simply, for each no-answer question in the qrels file:
    If the run has only one retrieved document and this document has the id of -1, 
     then the system receives a full score 
    Otherwise: the run it will receive a zero score for that question
    zero_answer_question_ids: the ids of the no-answer questions
    df_run: the run of the no-answer questions, which needs to be evaluated
    '''

    scores = []
    run_question_ids = df_run["qid"].values

    for qid in zero_answer_question_ids:
        if qid not in run_question_ids:
        # if this question does not exist in the run file, give it a zero credit
            scores.append(0)
        
        # select the rows where the docid equals to the current docid
        retrieved_doc_ids = df_run.loc[df_run["qid"] == qid, "docid"].values

        if len(retrieved_doc_ids) == 1 and retrieved_doc_ids[0] == "-1":
        # if there is only one retrieved document and this document has the id of -1,
        # then the system receives a full score
            scores.append(1)
        else: 
            #otherwise: it will receive a zero score
            scores.append(0)


    return scores


def evalaute_normal_questions(df_run, df_qrels):
    ''' Evaluate the performance for the normal questions
    df_run: the run dataframe of the normal questions
    df_qrels: the qrels dataframe of the normal questions
    The evaluation is performed using pytrec_eval tool (common tool for evaluating IR sysetems in python)
    '''
    # convert the qrels to a dictionary to match the pytrec_eval format
    qrels_dict = convert_to_dict(df_qrels,
                                column1='qid', 
                                column2='docid', 
                                column3='relevance')
    # convert the run into a dictionary  to match the pytrec_eval format
    run_dict = convert_to_dict(df_run,
                            column1='qid', 
                            column2='docid', 
                            column3='score')

    # initialize the evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, METRICS)
    # do the evaluation for each metric
    eval_res = evaluator.evaluate(run_dict)

    # extract the result values and put them in lists
    results = {}
    for metric in METRICS:
        metric_scores_list  = get_metric_list(results_dict=eval_res, metric=metric)
        results.update({metric: metric_scores_list})

    return results


def main(args):

    output_file = args.output
    qrels_file = args.qrels
    run_file = args.run
    format_check_passed = qqa23_sc.check_run(run_file)
    if not format_check_passed:
        return
    
    df_qrels = read_qrels_file(qrels_file)
    df_run = read_run_file(run_file)

    # select the zero answer questions from the qrel file
    zero_answer_question_ids = df_qrels.loc[df_qrels["docid"] == "-1", "qid"].values
    # get the qrels of the normal questions 
    df_qrels_normal_questions = df_qrels.loc[~df_qrels['qid'].isin(zero_answer_question_ids)]
    # divide the run into two dataframes, one contains the zero answer question
    df_zero_answer_questions = df_run[df_run['qid'].isin(zero_answer_question_ids)]
    # and the other contain the normal questions
    df_normal_questions = df_run[~df_run['qid'].isin(zero_answer_question_ids)]

    zero_answer_question_scores = evaluate_zero_answer_questions(zero_answer_question_ids, df_zero_answer_questions)
    normal_question_scores = evalaute_normal_questions(df_normal_questions, df_qrels_normal_questions)

    final_resutls = {}
    for metric in METRICS:
        metric_score_list = normal_question_scores[metric]
        metric_score_list.extend(zero_answer_question_scores)
        metric_score = np.mean(metric_score_list)
        final_resutls.update({metric: metric_score})
    
    df = pd.DataFrame([final_resutls])
    if output_file:
        df.to_csv(output_file, sep='\t', index=False)
        logger.info(f'Saved results to file: {output_file}')
    else:
        print(df.to_string(index=False))
    return final_resutls


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r', required=True,
                        help='run file with predicted scores from your model.\
                        Format: qid Q0 docid rank score tag')
    parser.add_argument('--qrels', '-q', required=True,
                        help='QRELS file with gold labels. Format: qid 0 docid relevance')
    parser.add_argument('--output', '-o',
                        help='Output file with metrics.\
                        If not specified, prints output in stdout.')
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    main(args)






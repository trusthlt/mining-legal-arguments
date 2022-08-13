#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import csv
from ast import literal_eval    
from collections import Counter
import numpy as np
from tabulate import tabulate
from multiTaskModel import compute_f1

id2label_argType = ['B-Distinguishing',
 'B-Einschätzungsspielraum',
 'B-Entscheidung des EGMR',
 'B-Konsens der prozessualen Parteien',
 'B-Overruling',
 'B-Rechtsvergleichung',
 'B-Sinn & Zweck Auslegung',
 'B-Subsumtion',
 'B-Systematische Auslegung',
 'B-Verhältnismäßigkeitsprüfung – Angemessenheit',
 'B-Verhältnismäßigkeitsprüfung – Geeignetheit',
 'B-Verhältnismäßigkeitsprüfung – Legitimer Zweck',
 'B-Verhältnismäßigkeitsprüfung – Rechtsgrundlage',
 'B-Vorherige Rechtsprechung des EGMR',
 'B-Wortlaut Auslegung',
 'I-Distinguishing',
 'I-Einschätzungsspielraum',
 'I-Entscheidung des EGMR',
 'I-Konsens der prozessualen Parteien',
 'I-Overruling',
 'I-Rechtsvergleichung',
 'I-Sinn & Zweck Auslegung',
 'I-Subsumtion',
 'I-Systematische Auslegung',
 'I-Verhältnismäßigkeitsprüfung – Angemessenheit',
 'I-Verhältnismäßigkeitsprüfung – Geeignetheit',
 'I-Verhältnismäßigkeitsprüfung – Legitimer Zweck',
 'I-Verhältnismäßigkeitsprüfung – Rechtsgrundlage',
 'I-Vorherige Rechtsprechung des EGMR',
 'I-Wortlaut Auslegung',
 'O']
label2id_argType = {}
for i, label in enumerate(id2label_argType):
    label2id_argType[label] = i
    
id2label_agent = ['B-Beschwerdeführer',
 'B-Dritte',
 'B-EGMR',
 'B-Kommission/Kammer',
 'B-Staat',
 'I-Beschwerdeführer',
 'I-Dritte',
 'I-EGMR',
 'I-Kommission/Kammer',
 'I-Staat',
 'O']
label2id_agent = {}
for i, label in enumerate(id2label_agent):
    label2id_agent[label] = i


def load_predictions(file):
    """
    Loads saved model predictions and returns them in a Dataframe.
    :param file: path of the predictions to be loaded
    :return: DataFrame of these predictions"""
    df = pd.read_csv(file, sep='\t', encoding='utf-8')
    df['Labels'] = df['Labels'].map(lambda x: literal_eval(x))
    df['Predictions'] = df['Predictions'].map(lambda x: literal_eval(x))
    df['Tokens'] = df['Tokens'].map(lambda x: literal_eval(x))
    return df


def compute_macro_f1(gold, pred, id2label):
    """
    Computes the Macro F1 Score over all classes.
    :param gold: the gold standard
    :param pred: the model predictions
    :return: the Macro F1 score"""
    f1s = [(tag, compute_f1(tag, gold, pred)) for tag in range(len(id2label))]
    
    all_f1s = [(id2label[idx], score) for idx, score in f1s]
    #print('F1 for each Class: ', all_f1s)
    
    f1_scores = [f1[1] for f1 in f1s]
    macro_f1 = np.sum(f1_scores) / len(f1_scores)
    #print('Macro F1: ', macro_f1)
    return np.around(macro_f1, decimals=4), all_f1s


# choose models to compare. can change models to own model. schema $DisplayName: $prediction_path_without_label_type_at_the_end

# dev set
files = {'LB1': 'predictions/val_preds/legal-bert-final_checkpoint-39820_bert', 'LB2': 'predictions/val_preds/legal-bert-final_checkpoint-47784_bert',
        'RBL1': 'predictions/val_preds/roberta-large-final_checkpoint-111482_roberta', 'RBL2': 'predictions/val_preds/roberta-large-final_checkpoint-143334_roberta',
        'FP13k1': 'predictions/val_preds/roberta-large-fp-13000_checkpoint-95556_roberta', 'FP13k2': 'predictions/val_preds/roberta-large-fp-13000_checkpoint-143334_roberta',
        'FP15k1': 'predictions/val_preds/roberta-large-fp-15000_checkpoint-143334_roberta', 'FP15k2': 'predictions/val_preds/roberta-large-fp-15000_checkpoint-159260_roberta'}

# test set
files = {'LB1': 'predictions/val_preds/legal-bert-final_checkpoint-39820_bert', 'LB2': 'predictions/legal-bert-final_checkpoint-47784_bert',
        'RBL1': 'predictions/roberta-large-final_checkpoint-111482_roberta', 'RBL2': 'predictions/roberta-large-final_checkpoint-143334_roberta',
        'FP13k1': 'predictions/roberta-large-fp-13000_checkpoint-95556_roberta', 'FP13k2': 'predictions/roberta-large-fp-13000_checkpoint-143334_roberta',
        'FP15k1': 'predictions/roberta-large-fp-15000_checkpoint-143334_roberta', 'FP15k2': 'predictions/roberta-large-fp-15000_checkpoint-159260_roberta'}

# test set art 3 (data distribution differs, only shown for the first model, distribution does not reflect data of second model
#files = {'Art. 3': 'predictions/article_3/roberta-large-fp-15000_checkpoint-143334_roberta', 'Best': 'predictions/roberta-large-fp-15000_checkpoint-143334_roberta'}


# initialize DataFrames (ArgType and Agent) with labels and frequency of them and sort by their frequency
for k,v in files.items():
    arg = load_predictions(v + '-argType.csv')
    agent = load_predictions(v + '-agent.csv')
    break

readable_labels_arg = arg['Labels'].map(lambda x: [id2label_argType[y] for y in x])
labels_arg = [label for sublist in readable_labels_arg for label in sublist]
freq_arg = Counter(labels_arg)
df_arg = pd.DataFrame(columns=['Label', 'Frequency', 'Percentage'])
df_arg = df_arg.append({'Label': 'Macro F1', 'Frequency': len(labels_arg), 'Percentage': 100}, ignore_index=True)
for l in freq_arg:
    df_arg = df_arg.append({'Label': l, 'Frequency': freq_arg[l], 'Percentage': np.around(freq_arg[l] / len(labels_arg) * 100, decimals=2)}, ignore_index=True)
df_arg = df_arg.sort_values('Frequency', ascending=False, ignore_index=True)

readable_labels_ag = agent['Labels'].map(lambda x: [id2label_agent[y] for y in x])
labels_ag = [label for sublist in readable_labels_ag for label in sublist]
freq_ag = Counter(labels_ag)
df_ag = pd.DataFrame(columns=['Label', 'Frequency', 'Percentage'])
df_ag = df_ag.append({'Label': 'Macro F1', 'Frequency': len(labels_ag), 'Percentage': 100}, ignore_index=True)
for l in freq_ag:
    df_ag = df_ag.append({'Label': l, 'Frequency': freq_ag[l], 'Percentage': np.around(freq_ag[l] / len(labels_ag) * 100, decimals=2)}, ignore_index=True)
df_ag = df_ag.sort_values('Frequency', ascending=False, ignore_index=True)


# for each model add f1 scores to the labels
for k,v in files.items():
    arg = load_predictions(v + '-argType.csv')
    agent = load_predictions(v + '-agent.csv')
    f1_arg, f1s_arg = compute_macro_f1(gold=arg.Labels.to_list(), pred=arg.Predictions.to_list(), id2label=id2label_argType)
    f1_ag, f1s_ag = compute_macro_f1(gold=agent.Labels.to_list(), pred=agent.Predictions.to_list(), id2label=id2label_agent)
    sort_f1s_arg = []
    d_arg = dict(f1s_arg)
    d_arg['Macro F1'] = f1_arg
    for l in df_arg.Label.tolist():
        sort_f1s_arg.append(np.around(d_arg[l] * 100, decimals=2))
    df_arg[k] = sort_f1s_arg
    sort_f1s_ag = []
    d_ag = dict(f1s_ag)
    d_ag['Macro F1'] = f1_ag
    for l in df_ag.Label.tolist():
        sort_f1s_ag.append(np.around(d_ag[l] * 100, decimals=2))
    df_ag[k] = sort_f1s_ag

# when comparing two models, also include relative change of first model compared to the second 
if len(df_arg.columns) == 5:
    diffs = []
    for i, row in df_arg.iterrows():
        diffs.append(- np.around(((1 - (row['Art. 3'] / (row['Best'] + 1e-10))) * 100), decimals=2))
    df_arg['Relative Change (%)'] = diffs

    diffs = []
    for i, row in df_ag.iterrows():
        diffs.append(- np.around(((1 - (row['Art. 3'] / (row['Best'] + 1e-10))) * 100), decimals=2))
    df_ag['Relative Change (%)'] = diffs


# display results
print('\n*****ArgType*****\n')
print(tabulate(df_arg, headers='keys', tablefmt='pretty', showindex=False))

print('\n*****Agent*****\n')
print(tabulate(df_ag, headers='keys', tablefmt='pretty', showindex=False))

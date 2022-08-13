#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from ast import literal_eval
from sklearn.metrics import confusion_matrix
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix

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
    

# name of cm -> path to predictions without the label type at the end
files = {'LEGAL-BERT': 'predictions/legal-bert-final_checkpoint-39820_bert', 
         'RoBERTa Large': 'predictions/roberta-large-final_checkpoint-111482_roberta', 
        'Further Pretraining for 13k steps of RoBERTa Large on legal data': 'predictions/roberta-large-fp-13000_checkpoint-95556_roberta', 
        'Further Pretraining for 15k steps of RoBERTa Large on legal data': 'predictions/roberta-large-fp-15000_checkpoint-143334_roberta'}

# plot cms
cmap = 'Oranges'
for k,v in files.items():
    arg = load_predictions(v + '-argType.csv')
    ag = load_predictions(v + '-agent.csv')

    # agent
    cm = confusion_matrix([l for sublist in ag['Labels'] for l in sublist], [p for sublist in ag['Predictions'] for p in sublist], labels=range(len(id2label_agent)))
    df_cm = pd.DataFrame(cm, index=id2label_agent, columns=id2label_agent)
    pretty_plot_confusion_matrix(df_cm, cmap=cmap, figsize=[11,11], title='Confusion Matrix ' + k + ' Agent', path='figures/confusion_matrices/cm_' + '_'.join(k.split()) + '_agent.png')

    # arg type
    cm = confusion_matrix([l for sublist in arg['Labels'] for l in sublist], [p for sublist in arg['Predictions'] for p in sublist], labels=range(len(id2label_argType)))
    df_cm = pd.DataFrame(cm, index=id2label_argType, columns=id2label_argType)
    pretty_plot_confusion_matrix(df_cm, cmap=cmap, figsize=[31,31], title='Confusion Matrix ' + k + ' Argument Type', path='figures/confusion_matrices/cm_' + '_'.join(k.split()) + '_argType.png')

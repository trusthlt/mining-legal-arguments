from sklearn.svm import SVC  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
from cassis import *
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


with open('gold_data/TypeSystem.xml', 'rb') as f:
    typesystem = load_typesystem(f)


def read_xmi(file, typesystem=typesystem):
    """
    Reads the data from an xmi file and returns the tokens, paragraphs and annotations.
    :param file: path to xmi file
    :param typesystem: typesystem of the xmi
    :return: tokens, paragraphs and annotations"""
    with open(file, 'rb') as f:
        cas = load_cas_from_xmi(f, typesystem=typesystem)
    tokens = cas.select('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token')
    #paragraphs = cas.select('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph')
    paragraphs = cas.select('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence')
    #sents = cas.select('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence')
    annotations = cas.select('webanno.custom.LegalArgumentation')
    return tokens, paragraphs, annotations


def read_data(path, cases=None):
    """
    Reads the data from all xmi files in the specified path and returns lists of the tokens, paragraphs and annotations.
    :param path: path to directory with the xmi files
    :param cases: dictionary of the cases
    :return:  dictionary of the cases with the xmi data added"""
    docs_token = []
    docs_paras = []
    docs_anno = []
    files = [f for f in os.listdir(path) if f.endswith('.xmi')]
    del files[files.index('001-67472.xmi')]
    del files[files.index('001-175007.xmi')]
    if not cases:
        cases = {}
        for file in files:
            cases[file[:-4]] = {}
    for f in files:
        data = read_xmi(os.path.join(path, f))
        cases[f[:-4]]['token_xmi'] = data[0]
        cases[f[:-4]]['para_xmi'] = data[1]
        cases[f[:-4]]['anno_xmi'] = data[2]
    return cases


def prepare_data(tokens, annotations):
    """
    Converts the xmi data into the tokens and corresponding BIO tags for the argType and agent.
    :param tokens: list of xmi tokens
    :param annotations: list of xmi annotations
    :return: list of tokens, list of argType BIO tags and list of agent BIO tags"""
    tokens_raw = [x.get_covered_text() for x in tokens]
    
    # id -> position in list of tokens so we can insert the bio tag at appropriate place
    lookup = dict()
    for i, token in enumerate(tokens):
        lookup[token.xmiID] = i
        
    bio_tags_args = len(tokens)*['O'] # O tag for non-arguments
    for anno in annotations:
        start = anno.begin
        end = anno.end
        for tok in tokens:
            # B tag for begin
            if tok.begin == start and anno.ArgType is not None:
                bio_tags_args[lookup[tok.xmiID]] = 'B-' + anno.ArgType
            # I tag for in between start and end(can't be == end because end is exclusive)
            elif tok.begin > start and tok.begin < end and anno.ArgType is not None:
                bio_tags_args[lookup[tok.xmiID]] = 'I-' + anno.ArgType
    
    # same for agent tags
    bio_tags_agent = len(tokens)*['O'] # O tag for non-arguments
    for anno in annotations:
        start = anno.begin
        end = anno.end
        for tok in tokens:
            # B tag for begin
            if tok.begin == start and anno.Akteur is not None:
                bio_tags_agent[lookup[tok.xmiID]] = 'B-' + anno.Akteur
            # I tag for in between start and end(can't be == end because end is exclusive)
            elif tok.begin > start and tok.begin < end and anno.Akteur is not None:
                bio_tags_agent[lookup[tok.xmiID]] = 'I-' + anno.Akteur
                
    return tokens_raw, bio_tags_args, bio_tags_agent


def paragraphed_tokens(tokens, paragraphs, shorten=True): 
    """
    Divides the tokens into our input units (paragraphs).
    :param tokens: list of xmi tokens
    :param annotations: list of xmi paragraphs
    :return: list of paragraphs"""    
    paragraphed_token = []
    for para in paragraphs:
        start = para.begin
        end = para.end
        para_toks = []
        for tok in tokens:
            if tok.begin >= start and tok.begin < end:
                para_toks.append(tok.get_covered_text())
        paragraphed_token.append(para_toks)
        # shorten files because argumentation starts only after "THE LAW"
        if shorten and para_toks == ['THE', 'LAW']:
            paragraphed_token = []
        if shorten and para_toks == ['AS', 'TO', 'THE', 'LAW']:
            paragraphed_token = []
            
    return paragraphed_token


def plot_coefficients(coef, feature_names, title='', top_features=5):
    #coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15,5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.barh(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.title(title, fontdict = {'fontsize' : 16})
    plt.xticks(fontsize=12)
    plt.yticks(np.arange(0, 0 + 2 * top_features), feature_names[top_coefficients], rotation=0, ha='right', fontsize=14)
    plt.tight_layout()
    plt.show()



list_argType = ['Distinguishing',
 'Einschätzungsspielraum',
 'Entscheidung des EGMR',
 'Konsens der prozessualen Parteien',
 'Overruling',
 'Rechtsvergleichung',
 'Sinn & Zweck Auslegung',
 'Subsumtion',
 'Systematische Auslegung',
 'Verhältnismäßigkeitsprüfung – Angemessenheit',
 'Verhältnismäßigkeitsprüfung – Geeignetheit',
 'Verhältnismäßigkeitsprüfung – Legitimer Zweck',
 'Verhältnismäßigkeitsprüfung – Rechtsgrundlage',
 'Vorherige Rechtsprechung des EGMR',
 'Wortlaut Auslegung']

list_agent = ['Beschwerdeführer', 'EGMR', 'Staat', 'Kommission/Kammer', 'Dritte']


# whether to recompute the dict containing the mapping from cases name to importance level(necessary with new data)
recompute_importance_dict = False
# whether to use already saved cases dict with everything (tokens, bio-tags ...) but the xmi data for feature creation (avoid unnecessary computation if using the same data)
use_precomputed = False
# whether to use the final dict with the feature values and the associated importance level 
recompute_feature_dict = False

if recompute_importance_dict:
    goldfiles = [f.split('.')[0] for f in os.listdir('gold_data/') if f.endswith('.xmi')]

    # path to scraped files with imoprtance information
    directory = 'ECHR-Scraper-master-echrscraper-rss_approach-03_all_cases_html/echrscraper/rss_approach/03_all_cases_html/'
    directories = [f for f in os.listdir(directory)]

    importance_dict = {}
    # match with gold data
    for d in directories:
        files = [f.split('.')[0] for f in os.listdir(os.path.join(directory, d)) if f.endswith('.json')]
        for match in set(goldfiles).intersection(set(files)):
            with open(os.path.join(directory, d, match + '.json'), 'r') as f:
                case = json.load(f)
            importance_dict[match] = case['results'][0]['columns']['importance']

    # original data used excluded the following files
    del importance_dict['001-67472']
    del importance_dict['001-175007']

    # save importance dict
    with open('gold_data/importance_dict.json', 'w') as f:
        json.dump(importance_dict, f)


if recompute_feature_dict:
    # recompute from scratch 
    if not use_precomputed:
        # read xmi data
        cases = read_data('gold_data/')
        
        # convert to tokens and tags for each file/case
        for k in cases.keys():
            cases[k]['token_raw'], cases[k]['bio_tags_args'], cases[k]['bio_tags_agent'] = prepare_data(cases[k]['token_xmi'], cases[k]['anno_xmi'])

        # get paragraphed tokens for each case
        for k in cases.keys():
            cases[k]['para_token_shortened'] = paragraphed_tokens(cases[k]['token_xmi'], cases[k]['para_xmi'])


        # also shorten ArgType and Agent tags
        for k,v in cases.items():
            skipped = len([item for sublist in v['para_token_shortened'] for item in sublist])
            cases[k]['bio_tags_args_shortened'] = cases[k]['bio_tags_args'][-skipped:]
            cases[k]['bio_tags_agent_shortened'] = cases[k]['bio_tags_agent'][-skipped:]

        # get paragraph representation for both tagsets
        for k,v in cases.items():
            start = 0
            paragraphed_labels_argType = []
            paragraphed_labels_agent = []
            for par in v['para_token_shortened']:
                end = start + len(par)
                paragraphed_labels_argType.append(v['bio_tags_args_shortened'][start:end])
                paragraphed_labels_agent.append(v['bio_tags_agent_shortened'][start:end])
                start = end
            cases[k]['para_args_shortened'] = paragraphed_labels_argType
            cases[k]['para_agent_shortened'] = paragraphed_labels_agent


        # add shortened tokens
        for k,v in cases.items():
            cases[k]['token_raw_shortened'] = v['token_raw'][-len(v['bio_tags_agent_shortened']):]

        # clean xmi data since it cannot be saved in json
        for k in cases.keys():
            cases[k]['token_xmi'] = ''
            cases[k]['para_xmi'] = ''
            cases[k]['anno_xmi'] = ''
    
        # save computation
        with open('gold_data/cases_features.json', 'w') as f:
            json.dump(cases, f)

    # use precomputed
    with open('gold_data/cases_features.json', 'r') as f:
        cases = json.load(f)

    # add xmi data
    cases = read_data('gold_data/', cases)

    # compute features (add new features here)
    for k,v in cases.items():
        if not v['anno_xmi']:
            print('No annotation in file ', k)
            no_annos.append(k)
            continue
        features = {}
        features['Doc Length'] = len(v['token_raw'])
        features['Fraction Argumentive Part'] = 1 - Counter(v['bio_tags_args'])['O'] / len(v['token_raw'])
        features['Shortened Doc Length'] = len(v['token_raw_shortened'])
        features['Shortened Fraction Argumentive Part'] = 1 - Counter(v['bio_tags_args_shortened'])['O'] / len(v['token_raw_shortened'])
        features['No. of Args'] = len(v['anno_xmi'])

        argTypes = [anno.ArgType for anno in v['anno_xmi']]
        agents = [anno.Akteur for anno in v['anno_xmi']]
        c_argTypes = Counter(argTypes)
        c_agents = Counter(agents)
        for arg in list_argType:
            #features[f'No. of {arg} Args'] = c_argTypes[arg]
            features[f'Fraction of {arg} Arg'] = c_argTypes[arg] / features['No. of Args']
        for agent in list_agent:
            #features[f'No. of {agent} Agents'] = c_agents[agent]
            features[f'Fraction of {agent} Agent'] = c_agents[agent] / features['No. of Args']
        features['Avg. Arg Length (Chars)'] = sum([anno.end - anno.begin for anno in v['anno_xmi']]) / len(v['anno_xmi'])
        cases[k]['features'] = features

     
    # extract features with importance level and save them
    df = pd.DataFrame(columns=list(cases['001-101152']['features'].keys()) + ['Importance'])
    y = []
    for k,v in cases.items():
        df = df.append(v['features'], ignore_index=True)
        y.append(int(importance_dict[k]))
    df['Importance'] = y
    
    df.to_csv('gold_data/importance_model_features.csv', encoding='utf-8', index=False, sep='\t')
    



visualize_feature_importance = True    

    



# read features with classes
df = pd.read_csv('gold_data/importance_model_features.csv', encoding='utf-8', sep='\t')
print('Distribution', Counter(df['Importance']))

X = df.drop('Importance', axis=1)  
y = df['Importance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=18)
print('Train Distribution', Counter(y_train))
print('Test Distribution', Counter(y_test))

# standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = [
  {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale'], 'degree': [2, 3, 4, 5, 6], 'kernel': ['poly']}
 ]

grid = GridSearchCV(SVC(max_iter=1000000), param_grid, refit=True, verbose=3, scoring='f1_macro')
grid.fit(X_train_scaled,y_train)

print('Best Grid Params: ', grid.best_params_)
print('Cross validation score of these params ', grid.best_score_)
preds = grid.predict(X_test_scaled)
print('Test scores')
print(classification_report(y_true=y_test, y_pred=preds))

if visualize_feature_importance:
    #for i, t in enumerate(['1 vs. 2', '1 vs. 3', '1 vs. 4', '2 vs. 3', '2 vs. 4', '3 vs. 4']):
        #plot_coefficients(grid.best_estimator_.coef_[i], X_train.columns, title=t)
    
    print('Average values for each improtance level:')
    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option('display.max_columns', 10)
    print(df.groupby('Importance').mean().transpose())




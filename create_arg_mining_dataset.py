from cassis import *
from collections import Counter
from prettytable import PrettyTable
import os 
import pandas as pd
import numpy as np

original_data = True

id2label_argType = ['O', 'B-Subsumtion', 'I-Subsumtion', 'B-Entscheidung des EGMR', 'I-Entscheidung des EGMR', 
            'B-Vorherige Rechtsprechung des EGMR', 'I-Vorherige Rechtsprechung des EGMR', 
           'B-Intitutionelle Argumente - Einschätzungsspielraum/Margin of Appreciation',
           'I-Intitutionelle Argumente - Einschätzungsspielraum/Margin of Appreciation', 
           'B-Intitutionelle Argumente - Distinguishing', 'I-Intitutionelle Argumente - Distinguishing', 
           'B-Intitutionelle Argumente - Overruling', 'I-Intitutionelle Argumente - Overruling', 
           'B-Verhältnismäßigkeitsprüfung - Angemessenheit/Erforderlichkeit', 
           'I-Verhältnismäßigkeitsprüfung - Angemessenheit/Erforderlichkeit',
           'B-Verhältnismäßigkeitsprüfung - Geeignetheit', 'I-Verhältnismäßigkeitsprüfung - Geeignetheit', 
           'B-Verhältnismäßigkeitsprüfung - Legitimer Zweck', 'I-Verhältnismäßigkeitsprüfung - Legitimer Zweck' , 
           'B-Verhältnismäßigkeitsprüfung - Rechtsgrundlage', 'I-Verhältnismäßigkeitsprüfung - Rechtsgrundlage', 
           'B-Auslegungsmethoden - Rechtsvergleichung', 'I-Auslegungsmethoden - Rechtsvergleichung',
           'B-Auslegungsmethoden - Sinn & Zweck', 'I-Auslegungsmethoden - Sinn & Zweck', 
           'B-Auslegungsmethoden - Systematische Auslegung', 'I-Auslegungsmethoden - Systematische Auslegung',
           'B-Auslegungsmethoden - Historische Auslegung', 'I-Auslegungsmethoden - Historische Auslegung',
           'B-Auslegungsmethoden - Wortlaut', 'I-Auslegungsmethoden - Wortlaut',
           'B-Konsens der prozessualen Parteien', 'I-Konsens der prozessualen Parteien']
label2id_argType = {}
for i, label in enumerate(id2label_argType):
    label2id_argType[label] = i

id2label_agent = ['O', 'B-Beschwerdeführer', 'I-Beschwerdeführer', 'B-EGMR', 'I-EGMR', 'B-Staat', 
                  'I-Staat', 'B-Kommission/Kammer', 'I-Kommission/Kammer',  'B-Dritte', 'I-Dritte']
label2id_agent = {}
for i, label in enumerate(id2label_agent):
    label2id_agent[label] = i

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


def show_distribution(labels, flatten=False):
    """
    Shows the distribution of the labels in the provided list as a table.
    :param labels: list of labels"""
    table = PrettyTable(['LABEL', 'FREQUENCY', 'PERCENTAGE'])
    table.float_format['PERCENTAGE'] = '.2'
    table.sortby = 'PERCENTAGE'
    table.reversesort = True
    if flatten:
        labels = [label for sublist in labels for label in sublist]
    freq = Counter(labels)
    for item in freq:
        table.add_row([item, freq[item], freq[item] / len(labels) * 100])
    print(table.get_string(end=51)) # change if using more than 25 Arg Types (25* 'B-' + 25* 'I-' + 'O' tag)


def read_data(path):
    """
    Reads the data from all xmi files in the specified path and returns lists of the tokens, paragraphs and annotations.
    :param path: path to directory with the xmi files
    :return:  list of tokens, list of paragraphs and annotations"""
    docs_token = []
    docs_paras = []
    docs_anno = []
    
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.xmi')]
    # when using exactly the data of the thesis, delete the two files with less than 5 annotations
    if original_data:
        del files[files.index(os.path.join(path, '001-67472.xmi'))]
        del files[files.index(os.path.join(path, '001-175007.xmi'))]
        
    for f in files:
        data = read_xmi(f)
        docs_token.append(data[0])
        docs_paras.append(data[1])
        docs_anno.append(data[2])
    return docs_token, docs_paras, docs_anno


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


def save_docs(path, filenames,  docs_paragraphed_tokens, docs_paragraphed_labels):
    """
    Saves the files as csv files with tokens and labels.
    :param path: directory to save at
    :param filenames: list of the names of each file
    :param docs_paragraphed_tokens: list with the paragraphed tokens for each file
    :param docs_paragraphed_labels: list with the paragraphed labels for each file"""
    assert len(filenames) == len(docs_paragraphed_tokens), 'Number of filenames should match the length of the document lists'
    for i, file in enumerate(filenames):
            df = pd.DataFrame({'tokens': docs_paragraphed_tokens[i], 'labels': docs_paragraphed_labels[i]})
            df.to_csv(os.path.join(path, file) , sep='\t', encoding='utf-8', index=False)



show_stats = True
subword_stats = True

# read files
docs_tok, docs_para, docs_anno = read_data('gold_data/')

# convert xmi data to tokens and tags for each file
docs_tok_raw = []
docs_bio_args = []
docs_bio_agents = []
for i in range(len(docs_anno)):
    data = prepare_data(docs_tok[i], docs_anno[i])
    docs_tok_raw.append(data[0])
    docs_bio_args.append(data[1])
    docs_bio_agents.append(data[2])

# get paragraphed tokens for each file
docs_para_tok = [paragraphed_tokens(tokens, paragraphs, shorten=True) for tokens, paragraphs in zip(docs_tok, docs_para)]

# also shorten ArgType and Agent tags
for i, doc in enumerate(docs_para_tok):
    skipped = len([item for sublist in doc for item in sublist])
    docs_bio_args[i] = docs_bio_args[i][-skipped:]
    docs_bio_agents[i] = docs_bio_agents[i][-skipped:]

# get paragraph representation for both tagsets
docs_para_argType = []
docs_para_agent = []
for i, paragraphed_tokens in enumerate(docs_para_tok):
    start = 0
    paragraphed_labels_argType = []
    paragraphed_labels_agent = []
    for par in paragraphed_tokens:
        end = start + len(par)
        paragraphed_labels_argType.append(docs_bio_args[i][start:end])
        paragraphed_labels_agent.append(docs_bio_agents[i][start:end])
        start = end
    docs_para_argType.append(paragraphed_labels_argType)
    docs_para_agent.append(paragraphed_labels_agent)


# save files
files = [f[:-4] + '.csv' for f in os.listdir('gold_data/') if f.endswith('.xmi')]
# when using exactly the data of the thesis, delete the two files with less than 5 annotations
if original_data:
    del files[files.index('001-67472.csv')]
    del files[files.index('001-175007.csv')]
save_docs('new_data/argType/', files, docs_para_tok, docs_para_argType)
save_docs('new_data/agent/', files, docs_para_tok, docs_para_agent)

# compute cutoffs for train, val, test in 80/10/10 split
trainindex = int(0.8 * len(docs_para_tok))
valindex = int(0.9 * len(docs_para_tok))

# save partitioned data
save_docs('new_data/train/argType/', files[:trainindex], docs_para_tok[:trainindex], docs_para_argType[:trainindex])  
save_docs('new_data/train/agent/', files[:trainindex], docs_para_tok[:trainindex], docs_para_agent[:trainindex])  

save_docs('new_data/val/argType/', files[trainindex:valindex], docs_para_tok[trainindex:valindex], docs_para_argType[trainindex:valindex])  
save_docs('new_data/val/agent/', files[trainindex:valindex], docs_para_tok[trainindex:valindex], docs_para_agent[trainindex:valindex])  

save_docs('new_data/test/argType/', files[valindex:], docs_para_tok[valindex:], docs_para_argType[valindex:]) 
save_docs('new_data/test/agent/', files[valindex:], docs_para_tok[valindex:], docs_para_agent[valindex:]) 

if show_stats:
    print('ArgTypes at Argument Level:')
    show_distribution([anno.ArgType for annotations in docs_anno for anno in annotations])
    print('Agents at Argument Level:')
    show_distribution([anno.Akteur for annotations in docs_anno for anno in annotations])
    print('BIO Tags of ArgTypes: ')
    show_distribution([argType for doc in docs_para_argType for argType in doc], flatten=True)
    print('BIO Tags of Agents: ')
    show_distribution([agent for doc in docs_para_agent for agent in doc], flatten=True)

    # further statistics 
    all_para_toks = [token for sublist in docs_para_tok for token in sublist]
    all_para_argType = [argType for sublist in docs_para_argType for argType in sublist]
    all_para_agent = [agent for sublist in docs_para_agent for agent in sublist]
    seq_len = [len(para) for sublist in docs_para_tok for para in sublist]

    doc_len = []
    for doc in docs_para_tok:
        length = 0
        for para in doc:
            length += len(para)
        doc_len.append(length)

    print('Sequence Statistics at Word Level:')
    table = PrettyTable(['SEQUENCE TYPE', 'LENGTH']) 
    table.float_format['LENGTH'] = '.1'
    table.add_row(['Document Min Length', min(doc_len)])
    table.add_row(['Document Max Length', max(doc_len)])
    table.add_row(['Document Mean Length', np.mean(doc_len)])
    table.add_row(['Document Median Length', np.median(doc_len)])
    table.add_row(['Paragraph Min Length', min(seq_len)])
    table.add_row(['Paragraph Max Length', max(seq_len)])
    table.add_row(['Paragraph Mean Length', np.mean(seq_len)])
    table.add_row(['Paragraph Median Length', np.median(seq_len)])
    print(table.get_string()) 


if show_stats and subword_stats:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    tokenizer.add_prefix_space=True

    docs_seq_len = []
    for doc in docs_para_tok:
        tokenized = tokenizer(doc, is_split_into_words=True)
        seq_lens = [len(seq) for seq in tokenized['input_ids']]
        docs_seq_len.append(seq_lens)

    print('Sequence Statistics at Roberta Tokenized (Subword) Level:')
    table = PrettyTable(['SEQUENCE TYPE', 'LENGTH']) 
    table.float_format['LENGTH'] = '.1'
    table.add_row(['Document Min Length', min([sum(doc_lens) for doc_lens in docs_seq_len])])
    table.add_row(['Document Max Length', max([sum(doc_lens) for doc_lens in docs_seq_len])])
    table.add_row(['Document Mean Length', np.mean([sum(doc_lens) for doc_lens in docs_seq_len])])
    table.add_row(['Document Median Length', np.median([sum(doc_lens) for doc_lens in docs_seq_len])])
    table.add_row(['Paragraph Min Length', min([para for doc_lens in docs_seq_len for para in doc_lens])])
    table.add_row(['Paragraph Max Length', max([para for doc_lens in docs_seq_len for para in doc_lens])])
    table.add_row(['Paragraph Mean Length', np.mean([para for doc_lens in docs_seq_len for para in doc_lens])])
    table.add_row(['Paragraph Median Length', np.median([para for doc_lens in docs_seq_len for para in doc_lens])])
    print(table.get_string()) 

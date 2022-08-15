# Mining Legal Arguments in Court Decisions &mdash; Data and software

Companion code to our arXiv preprint including the ``RMU:ECHR`` corpus.

Pre-print available at: https://arxiv.org/abs/2208.06178

Please use the following citation

```plain
@journal{Habernal.et.al.2022.arg,
  author = {Habernal, Ivan and Faber, Daniel and Recchia, Nicola and
            Bretthauer, Sebastian and Gurevych, Iryna and
            Spiecker genannt Döhmann, Indra and Burchard, Christoph}, 
  title = {{Mining Legal Arguments in Court Decisions}},
  journal = {arXiv preprint},
  year = {2022},
  doi = {10.48550/arXiv.2208.06178},
}
```

> **Abstract** Identifying, classifying, and analyzing arguments in legal discourse has been a prominent area of research since the inception of the argument mining field. However, there has been a major discrepancy between the way natural language processing (NLP) researchers model and annotate arguments in court decisions and the way legal experts understand and analyze legal argumentation. While computational approaches typically simplify arguments into generic premises and claims, arguments in legal research usually exhibit a rich typology that is important for gaining insights into the particular case and applications of law in general. We address this problem and make several substantial contributions to move the field forward. First, we design a new annotation scheme for legal arguments in proceedings of the European Court of Human Rights (ECHR) that is deeply rooted in the theory and practice of legal argumentation research. Second, we compile and annotate a large corpus of 373 court decisions (2.3M tokens and 15k annotated argument spans). Finally, we train an argument mining model that outperforms state-of-the-art models in the legal NLP domain and provide a thorough expert-based evaluation. All datasets and source codes are available under open lincenses at https://github.com/trusthlt/mining-legal-arguments

**Contact person**: Ivan Habernal, ivan.habernal@tu-darmstadt.de. https://www.trusthlt.org

*This repository contains experimental software and is published for the sole purpose of giving additional background details on the publication.*


## Data

The annotated data can be found in [gold_data](gold_data) which was preprocessed with [create_arg_mining_dataset.py](create_arg_mining_dataset.py) to create the model data found in [data](data) and show statistics about the dataset.

The predictions of our models can be found in [predictions](predictions) and can be used to create a confusion matrices with [create_confusion_matrix.py](create_confusion_matrix.py) or compare model with [compare_f1s.py](compare_f1s.py).

## Models

### Legal Argument Mining

#### Training

A model can be trained by running [multiTaskModel.py](multiTaskModel.py). Running `python multiTaskModel.py -h` shows commonly used command line options, further configuration of the tranformer training arguments can be found and changed at the end of the file.

#### Evaluation

After training a model, [evaluate.py](evaluate.py) can be used to evaluate the model performance and save its predictions.

### Importance Model

[importance_model.py](importance_model.py) can be used to train the importance model and evaluate it.


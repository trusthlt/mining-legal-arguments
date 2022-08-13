# mining-legal-arguments
Mining Legal Arguments in Court Decisions - Data and software

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


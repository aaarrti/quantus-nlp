#### Make sure to have [poetry](https://python-poetry.org/) installed

#### Check quantus_nlp/examples for usage example e.g.
     poetry run python quantus_nlp/examples/main.py ris

#### Lint 
     poetry run flake8 quantus_nlp

#### Format code
     poetry run black quantus_nlp

#### Training on TPU
     gcloud config set project linen-synthesis-353917
     gcloud auth login
     ./glcoud/create_tpu.sh <node name>
     python quantus_nlp/main.py train --tpu --no-jit

### Download dataset/saved_model
    dvc pull
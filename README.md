#### Make sure to have [poetry](https://python-poetry.org/) installed

#### Lint 
     poetry run flake8 quantus_nlp
#### Format code
     poetry run black quantus_nlp

#### Training on TPU
     gcloud config set project linen-synthesis-353917
     gcloud auth login
     ./glcoud/create_tpu.sh <node name>
     python quantus_nlp/main.py --tpu True train
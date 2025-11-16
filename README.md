For your code itself, the logical order of modules in this project is something like:

config.py – settings / paths / constants

data_loading.py – read raw data into a DataFrame

preprocessing.py – clean / filter / encode

features.py – feature engineering

models.py – model definitions

pipeline.py – glues together 2–5 in the order you want

evaluation.py – metrics, plots

tuning.py – hyperparameter search
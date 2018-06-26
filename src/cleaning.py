import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def read_data(path_data, files, categories):
    path_project = os.getcwd()
    datasets = []
    types = {w: 'category' for w in categories}
    for file in files:
        dataset = pd.read_csv(os.path.join(path_project, path_data, file), dtype=types)
        datasets.append(dataset)
    return datasets


def enc_cat(variable, test_set=False, encoders=None):
    if not(test_set):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(variable)
    else:
        label_encoder = encoders
        integer_encoded = label_encoder.transform(variable)
    return integer_encoded, label_encoder


def clean_data(dataset, na_action=False, test_set=False, encoders=None):
    dataset['Title'] = dataset.apply(lambda x: x['Name'].split(',')[1].strip().split('.')[0], axis=1)
    if test_set:
        dataset = dataset.replace({'Title': {'Dona': 'Mrs'}})
        dataset['Title'], encoder_title = enc_cat(dataset['Title'], test_set, encoders['Title'])
        dataset['Sex'], encoder_sex = enc_cat(dataset['Sex'], test_set, encoders['Sex'])
    else:
        dataset['Title'], encoder_title = enc_cat(dataset['Title'])
        dataset['Sex'], encoder_sex = enc_cat(dataset['Sex'])
    if na_action:
        dataset['Age'] = dataset['Age'].fillna(dataset.apply(
            lambda x: dataset[(~dataset.Age.isnull()) & (dataset.Title == x["Title"])].Age.median(), axis=1))
        dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
        dataset['Fare'] = dataset['Fare'].fillna(dataset.apply(
            lambda x: dataset[(~dataset.Fare.isnull()) & (dataset.Pclass == x["Pclass"])].Age.median(), axis=1))
    encoders_dic = {"Sex": encoder_sex, "Title": encoder_title}
    return dataset, encoders_dic

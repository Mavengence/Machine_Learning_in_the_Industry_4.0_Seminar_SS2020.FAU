import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def select_first_encounter(df):
    sorted_df = df.sort_values(by='encounter_id').copy()
    return sorted_df.loc[sorted_df.groupby('patient_nbr')['encounter_id'].head(1).index]


def select_model_features(df, categorical_col_list, numerical_col_list, PREDICTOR_FIELD, grouping_key='patient_nbr'):
    selected_col_list = [grouping_key] + [PREDICTOR_FIELD] + categorical_col_list + numerical_col_list   
    return df[selected_col_list]


def aggregate_dataset(df, grouping_field_list,  array_field):
    df = df.groupby(grouping_field_list)['encounter_id', array_field].apply(lambda x: x[array_field].values.tolist()).reset_index().rename(columns={0: array_field + "_array"}) 
    dummy_df = pd.get_dummies(df[array_field + '_array'].apply(pd.Series).stack()).sum(level=0)
    dummy_col_list = [x.replace(" ", "_") for x in list(dummy_df.columns)] 
    mapping_name_dict = dict(zip([x for x in list(dummy_df.columns)], dummy_col_list ) ) 
    concat_df = pd.concat([df, dummy_df], axis=1)
    new_col_list = [x.replace(" ", "_") for x in list(concat_df.columns)] 
    concat_df.columns = new_col_list
    return concat_df, dummy_col_list

    
def preprocess_df(df, categorical_col_list, numerical_col_list, predictor):
    temp_df = df.copy()
    imp = SimpleImputer(strategy='mean')
    temp_df[predictor] = df[predictor].astype(float)
    temp_df[categorical_col_list] = df[categorical_col_list].astype(str)
    temp_df[numerical_col_list] = imp.fit_transform(df[numerical_col_list])
    return temp_df.copy()


def patient_dataset_splitter(df, patient_key='patient_nbr'):
    test_percentage= 0.2
    val_percentage = 0.2
    PATIENT_ID_FIELD = patient_key
    
    train, test = train_test_split(df, test_size=test_percentage, stratify=df['readmitted'], random_state=42)
    train, validation = train_test_split(train, test_size=val_percentage, stratify=train['readmitted'], random_state=42)
    
    assert len(set(train[PATIENT_ID_FIELD].unique()).intersection(set(test[PATIENT_ID_FIELD].unique()))) == 0
    print("Test passed for patient data in only one partition")
    
    assert (train[PATIENT_ID_FIELD].nunique()  + test[PATIENT_ID_FIELD].nunique() + validation[PATIENT_ID_FIELD].nunique()) == df[PATIENT_ID_FIELD].nunique()
    print("Test passed for number of unique patients being equal!")
    
    assert len(train)  + len(test) + len(validation) == len(df)
    print("Test passed for number of total rows equal!")
    
    return train, validation, test


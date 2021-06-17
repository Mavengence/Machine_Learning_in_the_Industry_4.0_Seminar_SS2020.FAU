import pandas as pd


def check_null_values(df):
    missing_df = pd.DataFrame({'columns': df.columns, 
                            'percent_null': df.isnull().sum() * 100 / len(df), 
                           'percent_zero': df.isin([0]).sum() * 100 / len(df),
                            '?': df.isin(["?"]).sum() / len(df)})
    return missing_df.copy() 
    
def count_unique_values(df, cat_col_list):
    cat_df = df[cat_col_list].copy()
    val_df = pd.DataFrame({'columns': cat_df.columns, 'cardinality': cat_df.nunique()})
    return val_df.copy()


 
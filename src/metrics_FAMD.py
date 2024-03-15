import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix

def metric_fc(df_categ, true_df_categ): 
    """Computes the proportion of falsely classified individuals prer category.  

    Args:
        df_categ (pd.DataFrame): dataframe gathering only categorical variables
        true_df_categ (pd.DataFrame): ground truth dataframe gathering for categorical variables

    Returns:
        fc (float): rate of falsely classified individuals per category
    """
    fc = np.zeros(df_categ.shape[1])
    for i,col in enumerate(df_categ.columns.to_numpy()): 
        cm = confusion_matrix(true_df_categ[col], df_categ[col])
        fc[i] =(cm[1][0] + cm[0][1]) / cm.sum()
    return fc 


def compute_nrmse_weighted(dfpred, dftrue):
    """Computes de Normalized Root Mean Squared Error (NRMSE)

    Args:
        dfpred (pd.DataFrame): dataframe gathering for continuous variables
        dftrue (pd.DataFrame): ground truth dataframe gathering for continuous variables

    Returns:
        nrmse (float): NRMSE value for each variable 
    """
    # Convert DataFrames to numpy arrays
    df_true = dftrue.to_numpy()
    df_pred = dfpred.to_numpy()
    weights = df_true.sum(axis=0)
    std_df = df_true.std(axis=0)
   
    # Compute NRMSE numerator
    nrmse_numerator = np.sum(weights* ((df_true - df_pred)/std_df) ** 2)
    # Compute NRMSE denominator
    nrmse_denominator = np.sum(weights)
    # Compute NRMSE
    nrmse = np.sqrt(nrmse_numerator / nrmse_denominator)    
    return nrmse
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
# Compute weights for each variable based on the sum of true values
    # Convert DataFrames to numpy arrays
    weight_matrices = {}
    for column in dftrue.columns:
        weight_matrices[column] = 1 / np.abs(dftrue[column] - dftrue[column].mean() + 1e-6)

    weights = pd.DataFrame(weight_matrices, columns=dftrue.columns)
    if weights.sum().sum()==0: 
        weights += 1e-6
    std_df = dftrue.std(axis=0)
    weighted_squared_errors=0
    # Compute weighted sum of squared errors
    for column in dftrue.columns:
        weighted_squared_errors += ((((dftrue[column] - dfpred[column])/ std_df[column]) ** 2)*weights[column]).sum()

    # Compute NRMSE
    nrmse = np.sqrt(weighted_squared_errors / weights.sum().sum()) 
    return nrmse
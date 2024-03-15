import pandas as pd 
import numpy as np

def create_missingness(df, proba_non_missing):
    ''' Returns dataframe df with missing values
    
    Args: 
        df (pd.DataFrame): Dataframe to mask 
        proba_non_missing (float) : probability that a value is not missing

    Returns:
        data_missing_raw (pd.DataFrame): masked dataframe hence containing missing values 

    '''
    data_missing_raw = df.astype(object).mask(np.random.random(size=(df.shape))>proba_non_missing)
    return data_missing_raw


def encode_dummy_variables(df, cat_var_idx):
    """
    Encode dummy variables, returns the updated dataframe, the list of the dummy variables' names and the list of the number of different values in each categories
    Format of the dummy variable names : name_of_variable + _ + variable_value
    
    Args : 
        df (pd.DataFrame) 
        cat_var (pd.Index): index of the variables to encode into dummy variables

    Returns:
        df_dummy (pd.DataFrame): encoded dummy variables 
        dummy_var_idx (pd.Index): list of the dummy variables' names
        nb_values_per_cat (list): number of different values in each categories

    """

    dummy_var_idx = [] #list of the dummy variables' names
    df_dummy = df.copy()

    nb_values_per_cat = [] #list of the number of different values in each categories

    for var in cat_var_idx: 
        var_values = df_dummy[var].dropna().unique() #get the values of the categorical variable (excluding NA)
        nb_values_per_cat.append(len(var_values))

        for value in var_values :
            df_dummy[var + '_' + str(value)] = 1*(df_dummy[var] == value).mask(df_dummy[var].isna())
            dummy_var_idx.append(var + '_' + str(value))

    df_dummy = df_dummy.drop(cat_var_idx, axis = 1) 
    dummy_var_idx = pd.Index(dummy_var_idx)
    
    return df_dummy, dummy_var_idx, nb_values_per_cat

def generate_random(column, missing_indices_len):
    """Generate a random number between the minimum and maximum of a given variable

    Args:
        column (pd.Series): Column values

    Returns:
       random value (int): np.random.randint(low=column.min(), high=column.max()) 
    """
    return np.random.randint(low=column.min(), high=column.max()+1, size=missing_indices_len)

def random_imputation(df): 
    """Perform random imputation with random values per column

    Args:
        df (pd.DataFrame): _description_

    Returns:
        df_modif (pd.DataFrame): Imputed dataset
    """
    df_modif = df.copy()
    # Use the imputer to fill missing values in each column with different random values
    for col in df.columns:
        missing_indices = df[col].isna()
        missing_indices_len = len(np.where(missing_indices==True)[0])
        random_values = generate_random(df[col], missing_indices_len)
        df_modif.loc[missing_indices, col] = random_values
    return df_modif
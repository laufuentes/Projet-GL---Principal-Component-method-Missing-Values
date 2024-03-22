import pandas as pd 
import numpy as np

import pandas as pd 
import numpy as np


def create_dataset(n, S, K, cat, cat_idx, nb_of_cat_per_var, SNR = 1.0):
    """Generate dataset with continuous (float variables) and categorical variables (string variables) based on parameters.
    Args:
        n (int): Sample size.
        S (int): Underlying dimension.
        K (list of ints) : K[s] = number of times the variable s (s in {1,...,S}) is duplicated in the dataset 
        cat(int) : Number of categorical variables
        cat_idx (list of ints) : Indexes of the categorical variables 
        nb_of_cat_per_var (list of ints) : Number of categories for each categorical variable
        SNR (float) : Signal to Noise Ratio
    Returns:
        df (pd.Dataframe) : Generated dataset
    """

    #create S independant variables, drawn from a standard gaussian distribution
    df = np.random.normal(size = (S,n))

    nb_variables = S + sum(K) #number of variables expected in the final dataset
    data_shape = (nb_variables,n)

    #replicate each variable s (s in {1,...,S}) K_s times , to create correlated covariables:
    for s in range(S):
        for k in range(K[s]):
            df = np.vstack((df, df[s]))

    #add noise:
    mean_noise = 0
    std_noise = 1/SNR
    gaussian_noise = np.random.normal(mean_noise, std_noise, data_shape)
    df = df + gaussian_noise

    #create categorical variables :
    for i in range(cat): 
        idx_var = cat_idx[i] #index de la ième variable categorielle
        nb_of_cat = nb_of_cat_per_var[i] #nombre de categories différentes dans la ième variable categorielle
        df_i = df[idx_var] #selectionne la 

        #diviser df_i en nb_of_cat catégories différentes :
        #la méthode de division en catégories peut différer d'un datset a l'autre
        indices = np.arange(n)
        np.random.shuffle(indices)
        indices_cat = np.array_split(indices, nb_of_cat)

        for j in range(nb_of_cat):
            for ind in indices_cat[j]:
                df_i[ind] = j

        df[idx_var] = df_i

    df = pd.DataFrame(df.T)
    df[cat_idx] = df[cat_idx].astype(str)

    df.columns = df.columns.map(str)
    
    return df


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

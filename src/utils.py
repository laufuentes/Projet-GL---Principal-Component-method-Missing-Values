import pandas as pd 
import numpy as np

from src.metrics_FAMD import * 
from src.algorithms import *


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

    # Create S independant variables, drawn from a standard gaussian distribution
    df = np.random.normal(size = (S,n))

    nb_variables = S + sum(K) # Number of variables expected in the final dataset
    data_shape = (nb_variables,n)

    # Replicate each variable s (s in {1,...,S}) K_s times , to create correlated covariables:
    for s in range(S):
        for k in range(K[s]):
            df = np.vstack((df, df[s]))

    # Add noise:
    mean_noise = 0
    std_noise = 1/SNR
    gaussian_noise = np.random.normal(mean_noise, std_noise, data_shape)
    df = df + gaussian_noise

    # Create categorical variables :
    for i in range(cat): 
        idx_var = cat_idx[i] # Index of ith categorical variable
        nb_of_cat = nb_of_cat_per_var[i] # Number of different categories in ith categorical variable
        df_i = df[idx_var] 

        # Divide df_i in nb_of_cat different categories:
        # Categories partionioning can change from one dataset to another
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


def create_rare_df(f, n, S = 1, K = [4], SNR = 5):
    """Create dataset with rare categories, as described in section 3.3 of the paper

    Args:    
        f (float) : frequency of the rare categories
        n (int) : sample size
        S (int): Underlying dimension.
        K (list of ints) : K[s] = number of times the variable s (s in {1,...,S}) is duplicated in the dataset 
        SNR (float) : Signal to Noise Ratio

    Returns: 
        df_rare (pd.Dataframe) : Generated dataset with rare categories 
    """
    # Create df with continuous variables
    df_rare = create_dataset(n, S, K, 0, [], [], SNR = SNR)

    # Create the rare categorical variables, following the paper method :
    cat = 3 #number of categorical variables
    cat_idx = ["2","3","4"] #index of the categorical variables
    nb_of_cat_per_var = [3,3,3] #number of categories for each categorical variable


    # first cat variable :
    idx_var = cat_idx[0] 
    nb_of_cat = nb_of_cat_per_var[0]
    df_i = df_rare[idx_var] 
    indices = np.arange(n)
    np.random.shuffle(indices)
    indices_cat = np.array_split(indices, nb_of_cat)
    for j in range(nb_of_cat):
        for ind in indices_cat[j]:
            df_i[ind] = j
    df_rare[idx_var] = df_i


    # code the two linked rare categorical variables :
    nb_of_cat = nb_of_cat_per_var[1]
    df_1 = df_rare[cat_idx[1]]
    df_2 = df_rare[cat_idx[2]]
    np.random.shuffle(indices)
    nb_rare = int(f * n)
    indices_rare = indices[0:nb_rare]
    indices_non_rare = np.setdiff1d(indices, indices_rare)
    np.random.shuffle(indices_non_rare)
    for ind in indices_rare:
            df_1[ind] = 0
            df_2[ind] = 0

    indices_cat1 = np.array_split(indices_non_rare, nb_of_cat-1)
    for j in range(0,nb_of_cat-1):
        for ind in indices_cat1[j]:
            df_1[ind] = j+1
    df_rare[cat_idx[1]] = df_1

    np.random.shuffle(indices_non_rare)
    indices_cat2 = np.array_split(indices_non_rare, nb_of_cat-1)
    for j in range(0,nb_of_cat-1):
        for ind in indices_cat2[j]:
            df_2[ind] = j+1
    df_rare[cat_idx[2]] = df_2

    return df_rare




def compute_metrics(df, cat_idx, n_it, n_components, proba_non_missing):
    """ Computes the falsely classified rate and nrmse over a synthetic dataset for different probabilities of missingness.

    Args:
        df (pd.DataFrame): dataframe to impute
        n_it (int): maximum number of iterations for iFAMD convergence
        n_components (int): number of principal components for reconstruction 
        proba_non_missing (list of float) : List of probabilities that a value is not missing

    Returns:
        data_missing_raw (pd.DataFrame): Masked dataframe hence containing missing values 
    """
    # Categorical Variables :
    idx_k2 = pd.Index(cat_idx)
    # Continuous Variables
    idx_k1 = df.columns.difference(idx_k2)

    dict_dfs = {}
    for p in proba_non_missing: 
        df_missing = create_missingness(df, p)
        
        # Encode dummy variables in the dataframe and in the dataframe with missing values :
        df_missing_dummy, idx_j, nb_values_per_cat_df = encode_dummy_variables(df_missing, idx_k2)
        df_dummy = encode_dummy_variables(df, idx_k2)[0]
        dict_dfs.update({p:[idx_k1, idx_j, df_missing_dummy, df_dummy, nb_values_per_cat_df]})

    #IFAMD
    fc_rate = []
    nmrse = []

    for p,values in dict_dfs.items(): 
        k1, k_j, df_missing, df_true, nb_val_per_car = values
        C0_missing, Categ_missing = df_missing.isna()[k1].to_numpy(), df_missing.isna()[k_j].to_numpy()  
        
        #Computation of iterative FAMD
        ifamd_df = IterativeFAMDImputer(n_components=n_components, data=df_missing, k1=k1, k2=k_j, nb_values_per_cat = nb_val_per_car)
        ifamd_df.impute(n_it)
        df = ifamd_df.df

        # We encode categories into 0,1
        res = ifamd_df.df[ifamd_df.k2].copy()
        pos = 0
        for h in range (len(idx_k2)) :
            col = [idx_j[pos+i] for i in range (nb_values_per_cat_df[h])]
            res["max_value"] = ifamd_df.df[col].max(axis = 1)
            for value in col:
                res[value] = (res[value] == res["max_value"]).astype(int)
            pos += nb_values_per_cat_df[h]
        res = res[ifamd_df.k2] 
        #Compute metrics 
        fc_rate.append(metric_fc(res[Categ_missing], df_true[k_j][Categ_missing]))

        # For continuous variables: 
        nmrse.append(compute_nrmse_weighted(df[k1][C0_missing], df_true[k1][C0_missing]))

    fc_rate = np.array(fc_rate)
    nmrse = np.array(nmrse)

    return fc_rate, nmrse, df_missing.columns


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

    dummy_var_idx = [] # List of the dummy variables' names
    df_dummy = df.copy()

    nb_values_per_cat = [] # List of the number of different values in each categories

    for var in cat_var_idx: 
        var_values = df_dummy[var].dropna().unique() # Get the values of the categorical variable (excluding NA)
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

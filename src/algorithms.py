import pandas as pd 
import numpy as np
from src.utils import * 

class FAMD:
    def __init__(self, data, k1, k2, nb_values_per_cat, n_components=2):
        """Initialisation

        Args:
            n_components (int, optional): number of components to use for reconstruction. Defaults to 2.
            data (pd.DataFrame, optional): Dataset to use for imputation. Defaults to None.
            k1 (pd.Index): columns of continuous variables 
            k2 (pd.Index): columns of categorical variables 
            nb_values_per_cat (list): constains the number of categories per variable (length equals to number of categorical variables)
        """
        self.nb_cat = len(nb_values_per_cat)
        self.n_components = n_components
        self.k1 = k1
        self.k2 = k2
        self.df = data
        self. J = len(k1) + len(k2) 
        self.df_C0 = data[self.k1]
        self.df_categ = data[self.k2]

    def data_concat(self): 
        """ Redefines the whole dataset (df) after updating df_C0 (continuous variables) and df_categ (categorical variables) with new imputation"""
        self.df = pd.DataFrame(np.concatenate((self.df_C0, self.df_categ), axis=1), columns=np.concatenate((self.k1.tolist(), self.k2.tolist())))
        pass

    def ponderation_gsbs(self): 
        """Weighting step specialized in the gsbs dataset: 
        This function updates: 
        - the standard deviation values (sj) for continous variables
        - proportion for categorical variables (pj) 
        according to a new imputation update. 
        """
        self.df_C0 = self.df[self.k1] # redefines df_C0 with updated df
        self.sj = self.df_C0.std(axis=0).to_numpy()

        self.df_categ = self.df[self.k2]
        print(self.df_categ)
        self.sqrt_pj = np.sqrt((self.df_categ.sum(axis=0)).to_numpy() / self.df_categ.shape[0].to_numpy())
        res = self.df_categ.copy()

        for h in range(3):
            col = [self.df_categ.columns[h],self.df_categ.columns[h+3]]
            somme = self.df_categ[col].sum(axis=1)
            for j in range(self.df.shape[0]):
                res.loc[j, col] = self.df_categ[col].iloc[j]/somme[j]
        self.df_categ = res
        self.data_concat()
        pass 

    def DM(self):
        """Function to define D and M according to current values"""

        # Computation of D_sigma
        self.D = np.diag(np.concatenate((self.sj,self.sqrt_pj)))
        D_moins_sqrt = np.sqrt(np.linalg.inv(self.D))
        
        # Computation of XD_sigma^(-0.5)
        self.XD_moins_sqrt = np.dot(self.df,D_moins_sqrt)
        
        # Computation of M
        self.M = self.XD_moins_sqrt.mean(axis=0)
        pass 
        
    def step3(self): 
        """Performs the third step from FAMD algorithm: 
        - Update D, M 
        - Computes the SVD on self.XD_moins_sqrt - self.M

        Returns:
            Z_p (pd.DataFrame): reconstructed version of (self.XD_moins_sqrt - self.M) according to self.n_components 
        """
        self.DM()
        U, S, Vt = np.linalg.svd(self.XD_moins_sqrt - self.M)

        # Computes the renormalized version of iFAMD by regularizing the diagonal terms of S from SVD 
        sigma2 = (1/(self.J - self.nb_cat - self.n_components))* np.array(S**2)[0:self.n_components]
        s = (np.array(S**2)[0:self.n_components] - sigma2 )/ np.array(S)[0:self.n_components]

        #Reconstruction into a lower dimension 
        Z_p = pd.DataFrame(U[:,:self.n_components]@np.diag(s)@Vt[:self.n_components,:], columns=self.df.columns, index=self.df.index)
        return Z_p
    
    def run_famd(self, verbose=False):
        """Function that runs the whole FAMD algorithm

        Returns:
            Z_p (pd.DataFrame): reconstructed version of (self.XD_moins_sqrt - self.M) according to self.n_components 
        """
        # Step 1: Initialization 
            # As we work on pre-processed data, we skip this step 

        # Step 2: Weighting 
        self.ponderation_technique()

        #Step 3: PCA Computation
        # Compute terms D, XD^(-0.5) and M 
        Z_p = self.step3()
        return Z_p
    

class IterativeFAMDImputer(FAMD):
    def __init__(self, data, k1, k2, nb_values_per_cat, n_components=2):
        super().__init__(data, k1, k2, nb_values_per_cat)
        """Initialization
        Args:
            n_components (int, optional): number of components to use for reconstruction. Defaults to 2.
            data (pd.DtaFrame, optional): Dataset to use for imputation. Defaults to None.
            k1 (pd.Index): columns of continuous variables 
            k2 (pd.Index): columns of categorical variables 
            nb_values_per_cat (list): constains the number of categories per variable (length equals to number of categorical variables)
        """
        self.nb_cat = len(nb_values_per_cat) # number of categories (not including dummies)
        self.nb_values_per_cat = nb_values_per_cat # list of the number of different values in each categories
        self.n_components = n_components

    def inital_impute(self):
        """Initial imputation: 
        - continuous variables: mean/variable
        - categories: proportion/category
        """

        #For continuous variables
        Ximp_C0 = self.df_C0.copy()
        for c in self.df_C0.columns.to_numpy(): 
            Ximp_C0[c] = Ximp_C0[c].fillna(self.df_C0[c].mean())  
        self.sj = Ximp_C0.std(axis=0).to_numpy()
        self.df_C0 = Ximp_C0 #/self.sj

        # For categprical variables 
        Ximp_categ = self.df_categ.copy()
        Ximp_categ =Ximp_categ.fillna((self.df_categ.sum(axis=0)/self.df_categ.shape[0]).pow(1/2))  
        res = Ximp_categ.copy() #/ (Ximp_categ.sum(axis=0)/Ximp_categ.shape[0]).pow(1/2)

        pos = 0 # to get the position of the first dummy variable of one categorical variable
        for h in range(self.nb_cat):
            col = [Ximp_categ.columns[pos+i] for i in range (self.nb_values_per_cat[h])]
            pos += self.nb_values_per_cat[h]
            somme = Ximp_categ[col].sum(axis=1)
            for j in range(self.df.shape[0]):
                res.loc[j, col] = Ximp_categ[col].iloc[j]/somme[j]
        self.sqrt_pj = (res.sum(axis=0)/self.df_categ.shape[0]).pow(1/2)
        self.df_categ = res
        self.data_concat()
        pass 

    def ponderation_gsbs(self): 
        """Weighting function specialized on the gsbs dataset
        This function updates: 
        - the standard deviation values (sj) for continous variables, 
        - proportion for categorical variables (pj) 
        according to a new imputation update. 
        """

        self.df_C0 = self.df[self.k1] # redefines df_C0 according to current df
        self.sj = self.df_C0.std(axis=0).to_numpy()

        self.df_categ = self.df[self.k2]  # redefines df_categ according to current df
        self.sqrt_pj = np.sqrt(self.df_categ.sum(axis=0)/self.df_categ.shape[0]).to_numpy()

        self.data_concat() # update of the whole dataset (df)

        
    def impute(self, n_it, tol=1e-4,verbose=False):
        """Runs the whole imputation process

        Args:
            n_it (int): number of iterations to run the algorithm
            tol (float, optional): Threshold to define early stopping criteria. Defaults to 1e-4.
            verbose (bool, optional): Defaults to False.

        Returns:
            self.df (pd.DataFrame): Imputed dataset after algorithm convergence 
        """
        # Initialization 
        idx_NA = 1- self.df.isna().astype(int).to_numpy() # 1 if obs 0 otherwise

        #Initial imputation 
        self.inital_impute()
        self.data_concat() # define the whole dataset with imputed variables (df_C0, df_categ)

        diff = np.inf
        last_chap = np.inf*np.ones_like(idx_NA)
        i= 0
        while i < n_it and diff > tol: 
            self.ponderation_gsbs()
            Z_p = self.step3() #Updating D, M already inside

            # Computation of reconstructed X for imputed values 
            X_chap = (Z_p + self.M)@np.sqrt(self.D) 

            # Redefinition of dataframe with new imputed values
            df = pd.DataFrame(idx_NA*(self.df).to_numpy() + + (1- idx_NA)*X_chap.to_numpy(), columns=self.df.columns)
            self.df= df # Update df in self

            # Computation of convergence criteria 
            diff = ((X_chap - last_chap)**2).mean().mean()
            last_chap = X_chap
            i += 1 
        if i < n_it: 
            print('Converged in', i)
        else: 
            print('Maximum iterations reached')    
        return self.df 

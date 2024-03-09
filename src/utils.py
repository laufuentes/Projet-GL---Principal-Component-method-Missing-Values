import pandas as pd 
import numpy as np

class FAMD:
    def __init__(self, n_components=2, data=None, k1=None, k2=None):
        """Initialisation

        Args:
            n_components (int, optional): _description_. Defaults to 2.
            data (_type_, optional): _description_. Defaults to None.
        """
        self.n_components = n_components
        self.k1 = k1
        self.k2 = k2
        self.df_C0 = data[k1]
        self.df_categ = data[k2]

    def process_categ(self): 
        dummies = pd.get_dummies(self.df_categ)
        self.df_categ = dummies 
        self.k2 = dummies.shape[1]


    def df(self): 
        """Redefinition des données à chaque fois que l'on change df_C0 et df_categ"""
        self.df = np.concatenate((self.df_C0, self.df_categ), axis=1)
        pass

    def ponderation(self): 
        self.sj = self.df_C0.std(axis=0).to_numpy()
        new_df_C0 = self.df_C0 / self.sj

        self.sqrt_pj = np.sqrt(self.df_categ.sum(axis=0)/self.df_categ.shape[0]).to_numpy()
        new_df_categ = self.df_categ / self.sqrtpj

        self.df_C0 = new_df_C0
        self.df_categ = new_df_categ / new_df_categ.sum(axis=0) ##TODO: verifier si la somme doit faire 1!!!
        self.df()

    def DM(self):
        """Function pour definir D et M à partir des valeur des données stockées en self"""
        # self.df() à laisser ? 
        # Calcul de D_sigma
        self.D = np.diag(np.concatenate((self.sj,self.sqrt_pj)))
        D_moins_sqrt = np.sqrt(np.linalg.inv(np.diag(self.D)))
        
        #Calcul de XD_sigma^(-0.5)
        self.XD_moins_sqrt = np.dot(self.df,D_moins_sqrt)
        
        #Calcul de M
        self.M = self.XD_moins_sqrt.mean(axis=0)
        pass 
        
    def run_famd(self,verbose=False, preprocessed=True):
        # Step 1: Initialisation 
        if not preprocessed: 
            self.process()
        
        # Step 2: Pondération 
        self.ponderation()

        #Step 3: Mise en place ACP
        # On calcule les termes D, XD^(-0.5) et M 
        self.DM()
        U, S, Vt = np.linalg.svd(self.XD_moins_sqrt - self.M)

        #Reconstruction dans une plus petite dimension
        Z_p = pd.DataFrame(U[:,:self.n_components]@ np.diag(S)[:self.n_components,:self.n_components] @Vt[:self.n_components,:], columns=Ximp.columns, index=Ximp.index)

        return Z_p
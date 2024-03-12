import pandas as pd 
import numpy as np

class FAMD:
    def __init__(self, data, k1, k2, n_components=2):
        """Initialisation

        Args:
            n_components (int, optional): _description_. Defaults to 2.
            data (_type_, optional): _description_. Defaults to None.
        """
        self.n_components = n_components
        self.k1 = k1
        self.k2 = k2
        self.df = data
        self.df_C0 = data[self.k1]
        self.df_categ = data[self.k2]

    def process_categ(self): 
        dummies = pd.get_dummies(self.df_categ)
        self.df_categ = dummies 
        self.k2 = dummies.shape[1]


    def data_concat(self): 
        """Redefinition des données à chaque fois que l'on change df_C0 et df_categ"""
        self.df = pd.DataFrame(np.concatenate((self.df_C0, self.df_categ), axis=1), columns=np.concatenate((self.k1.tolist(), self.k2.tolist())))
        pass

    def ponderation(self): 
        self.df_C0 = self.df[self.k1] # redefini df_C0 avec le df actuel
        self.sj = self.df_C0.std(axis=0).to_numpy()
        new_df_C0 = self.df_C0 / self.sj
        self.df_C0 = new_df_C0 # redefini df_categ avec le df actuel

        self.df_categ = self.df[self.k2]
        self.sqrt_pj = np.sqrt(self.df_categ.sum(axis=0)/self.df_categ.shape[0]).to_numpy()
        new_df_categ = self.df_categ / self.sqrt_pj #mise a jour de df_C0
        self.df_categ = new_df_categ / new_df_categ.sum(axis=0) ##TODO: verifier si la somme doit faire 1!!!

        self.data_concat() # mise a jour de df_categ

    def DM(self):
        """Function pour definir D et M à partir des valeur des données stockées en self"""
        # self.df() à laisser ? 
        # Calcul de D_sigma
        self.D = np.diag(np.concatenate((self.sj,self.sqrt_pj)))
        D_moins_sqrt = np.sqrt(np.linalg.inv(self.D))
        
        #Calcul de XD_sigma^(-0.5)
        self.XD_moins_sqrt = np.dot(self.df,D_moins_sqrt)
        
        #Calcul de M
        self.M = self.XD_moins_sqrt.mean(axis=0)
        pass 
        
    def step3(self): 
        self.DM()
        U, S, Vt = np.linalg.svd(self.XD_moins_sqrt - self.M)

        #Reconstruction dans une plus petite dimension
        Z_p = pd.DataFrame(U[:,:self.n_components]@ np.diag(S)[:self.n_components,:self.n_components] @Vt[:self.n_components,:], columns=self.df.columns, index=self.df.index)
        return Z_p
    
    def run_famd(self,verbose=False, preprocessed=True):
        # Step 1: Initialisation 
        if not preprocessed: 
            self.process()
        
        # Step 2: Pondération 
        self.ponderation()

        #Step 3: Mise en place ACP
        # On calcule les termes D, XD^(-0.5) et M 
        Z_p = self.step3()
        return Z_p
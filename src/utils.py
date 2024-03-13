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

    def ponderation_gsbs(self): 
        self.df_C0 = self.df[self.k1] # redefini df_C0 avec le df actuel
        self.sj = self.df_C0.std(axis=0).to_numpy()

        self.df_categ = self.df[self.k2]
        self.sqrt_pj = np.sqrt(self.df_categ.sum(axis=0)/self.df_categ.shape[0]).to_numpy()
        res = self.df_categ.copy()
        for h in range(3):
            col = [self.df_categ.columns[h],self.df_categ.columns[h+3]]
            somme = self.df_categ[col].sum(axis=1)
            for j in range(self.df.shape[0]):
                res.loc[j, col] = self.df_categ[col].iloc[j]/somme[j]
        self.df_categ = res
        self.data_concat()# mise a jour de df_categ
        pass 

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
    
    def run_famd(self,ponderation_technique, verbose=False, preprocessed=True):
        # Step 1: Initialisation 
        if not preprocessed: 
            self.process()
        
        # Step 2: Pondération 
        self.ponderation_technique()

        #Step 3: Mise en place ACP
        # On calcule les termes D, XD^(-0.5) et M 
        Z_p = self.step3()
        return Z_p
    



class IterativeFAMDImputer(FAMD):
    def __init__(self, data, k1, k2, n_components=2):
        super().__init__(data, k1, k2)
        """Initialisation
        Args:
            n_components (int, optional): _description_. Defaults to 2.
            data (_type_, optional): _description_. Defaults to None.
        """
        self.n_components = n_components
            
    def inital_impute(self, data):
        #Pour les variables continues
        self.sj = self.df_C0.std(axis=0).to_numpy()
        Ximp_C0 = self.df_C0.copy()
        for c in self.df_C0.columns.to_numpy(): 
            Ximp_C0[c] = Ximp_C0[c].fillna(self.df_C0[c].mean())  
        self.df_C0 = Ximp_C0

        # Pour les variables catégorielles 
        self.sqrt_pj = np.sqrt(self.df_categ.sum(axis=0)/self.df_categ.shape[0]).to_numpy()
        Ximp_categ = self.df_categ.copy() 
        Ximp_categ =Ximp_categ.fillna(np.sqrt(self.df_categ.sum(axis=0)/self.df_categ.shape[0]))  
        res = Ximp_categ.copy()
        for h in range(3):
            col = [Ximp_categ.columns[h],Ximp_categ.columns[h+3]]
            somme = Ximp_categ[col].sum(axis=1)
            for j in range(self.df.shape[0]):
                res.loc[j, col] = Ximp_categ[col].iloc[j]/somme[j]
        self.df_categ = res
        self.data_concat()
        pass 

    def ponderation_gsbs(self): 
        self.df_C0 = self.df[self.k1] # redefini df_C0 avec le df actuel
        self.sj = self.df_C0.std(axis=0).to_numpy()

        self.df_categ = self.df[self.k2]
        self.sqrt_pj = np.sqrt(self.df_categ.sum(axis=0)/self.df_categ.shape[0]).to_numpy()
        res = self.df_categ.copy()
        for h in range(3):
            col = [self.df_categ.columns[h],self.df_categ.columns[h+3]]
            somme = self.df_categ[col].sum(axis=1)
            for j in range(self.df.shape[0]):
                res.loc[j, col] = self.df_categ[col].iloc[j]/somme[j]
        self.df_categ = res
        self.data_concat() # mise a jour de df_categ

        
    def impute(self, n_it, tol=1e-4,verbose=False):
        # Initialisation 
        idx_NA = 1- self.df.isna().astype(int).to_numpy() # 1 si obs 0 sinon
        #Initial imputation 
        self.inital_impute(self.df)
        self.data_concat() # on construit les données avec df_C0 et df_categ remplis 

        diff = np.inf
        last_chap = np.inf*np.ones_like(idx_NA)
        i= 0
        while i < n_it and diff > tol: 
            self.ponderation_gsbs()
            Z_p = self.step3() #Updating D, M already inside
            X_chap = (Z_p + self.M)@np.sqrt(self.D)
            df = pd.DataFrame(idx_NA*(self.df).to_numpy() + + (1- idx_NA)*X_chap.to_numpy(), columns=self.df.columns)
            self.df= df
            diff = ((X_chap - last_chap)**2).mean().mean()
            last_chap = X_chap
            i +=1 
        if i < n_it: 
            print('Converged in', i)
        else: 
            print('Maximum iterations reached')    
        return self.df 
# A principal components method to impute missing values for mixed data

## Project: Guidelines in Statistical Learning, M2 Mathematics and Artificial Intelligence, Paris-Saclay University 

### Laura Fuentes and Ambre Adjevi

#### 1- Project overview

Welcome to our project repository for the course "Guidelines in Statistical Learning"! We are Laura Fuentes and Ambre Adjevi, M2 students in Mathematics and Artificial Intelligence at Paris-Saclay University. For our final project, we've focused on the paper titled "A Principal Components Method to Impute Missing Values for Mixed Data" [1].

When a dataset contains missing values, traditional statistical learning methods cannot be directly applied. Imputation, which involves substituting artificial values for the missing ones, addresses this issue by rendering datasets complete and analyzable. While techniques exist for imputing continuous or categorical data, when dealing with mixed data, these methods become inapplicable. Thus, there is a need to develop techniques specifically tailored to mixed data, contributing to the expanding literature on this subject. The authors propose an iterative imputation algorithm based on the Factorial Analysis for Mixed Data (FAMD) technique.

#### 2- Implementation details 

We have provided a comprehensive report summarizing the paper and have implemented the algorithms and metrics introduced in the article (\src). The primary goal of our implementation is to test the algorithm's properties presented in the article on both synthetic and real datasets [2].

##### 2.1- Testing the algorithm's properties with simulations

We offer four notebooks to facilitate understanding of the methodology and progressively assess the algorithm's properties:

> 1-Generate_datasets

This notebook demonstrates how we generated our synthetic dataset. It provides a "hello world!" example and outlines the process of creating synthetic datasets for testing the algorithm's properties.

> 2-Function_Familiarization_Guide.ipynb

This notebook offers a gentle introduction to computing the algorithm. It begins by creating a synthetic dataset, then processes the dataset by injecting missing values and encoding dummy variables. Once the dataset is prepared, it computes the regularized version of iterative FAMD. Finally, it evaluates the performance of the imputation technique by comparing the ground truth to the algorithm's outputs.

> 3-Properties_synthetic_dataset.ipynb
This notebook aims to test the properties of the regularized iterative FAMD algorithm as presented in the article. For each property, we conducted 200 simulations, creating datasets tailored to the specific property, running regularized iterative FAMD, and computing the metrics. The resulting figures can be found in 'images/synthetic'.


> 4-Properties_real_dataset.ipynb 
Similar to the previous notebook, this one evaluates some of the algorithm's properties on a real dataset (GBSG2) [2]. The resulting figures are saved in 'images/gbsg2'.


##### 2.2- Package requirements

To run this experiment several packages may be required. We propose to create a new conda environment taxi_advi by copy-pasting the following command on the terminal:
```
# Create a new Conda environment
conda create -n ifamd python=3.8

# Activate the Conda environment
conda activate ifamd

# Install the required packages
conda install ipykernel=6.29.3 ipython=8.22.1 matplotlib=3.8.3 numpy=1.26.4 pandas=2.2.1 scikit-learn=1.4.1.post1
```

Finally, select the ifamd conda environemnt just created as each notebook's kernel 


#### 3- References

**[1]**  Audigier V., Husson F. & Josse J. (2013), A principal components method to impute missing values for mixed data

**[2]** Peters A, Hothorn T (2012) ipred: Improved Predictors. URL http://CRAN.R-project.org/package=ipred, r package version 0.9-1
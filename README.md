# gbdt_python

This is python-version gdbt written by Yang Li(gmlyytt@outlook.com).

# envrionment

python 3.7

# how to run

python gbdt.py

# content 

GBDT is a cascaded regressor, which is built based on many base regressor, 
such as linear regressor or decison tree, etc.

We designed two base regressors(linear regressor and decision tree) to show the differences. 

We will show the theory and implementation of 2 basic part of our code to help you understand our project. 

- **base regressor**

    - decision tree
    
        - theory 
        
            The decision tree here is Cart regressor tree, 
            please click [here](https://en.wikipedia.org/wiki/Decision_tree_learning) to learn more.
        
        - implementation
        
            The base class is DecisionTree. And CartRegressor is a derived class 
            of DecisionTree, which is helpful for scalability. 
            
            At first, we need to fit the model and that means we should create the tree structure.
        
        
        

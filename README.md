# Heart-Disease-Prediction
 # This Project is about Heart Disease Prediction with python machine Learning

   # work flow  

   # 1. import the heart data set:
            (it has several health parameters and import all the dependencies)
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            
            
   # 2. Data pre processing:
              (need to process the dataset to make it fit and compatible for machine learning algorithms to learn)
      
   # 3. Train Test Split:
              ( after processing we need to split our dataset into training and testing part.i.e, split the features and 
                targets)
                X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.2,stratify=Y, random_state=2 )
      
   # 5.Model Training:
        --> (we are using Logistic Regression model because the uses case is binary classification we are going to 
              classify weather the person has a heart disease or not i.e., (Yes or No) / (1 or 0) kind of questions.)
        --> (once we train this logistic regression model with our training data we will do some evaluation on our model to 
              check its performance so, after that we will get a trained logistic regression model, to this model we feed 
              new data, our model predict weatcher the person has heart disease or not.)
              
  # 6. Model Evaluation:
         --> check the accuracy score for both training and testing data

  # 7.Building a predictive System:
         --> now we have to predict the input data and check the result..
              if the result is  1  indicate person has heart disease
              if the result is  0  indicate person doesn't have heart disease.
                
           


                                  
  

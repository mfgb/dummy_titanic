## Hands-on exercise

In this exercise, we will create a model together. It is a very simple example but it will give you an idea of how a data science team works.

### Data description

- PassengerId -- A numerical id assigned to each passenger.
- Survived -- Whether the passenger survived (1), or didn't (0). We'll be making predictions for this column.
- Pclass -- The class the passenger was in -- first class (1), second class (2), or third class (3).
- Name -- the name of the passenger.
- Sex -- The gender of the passenger -- male or female.
- Age -- The age of the passenger. Fractional.
- SibSp -- The number of siblings and spouses the passenger had on board.
- Parch -- The number of parents and children the passenger had on board.
- Ticket -- The ticket number of the passenger.
- Fare -- How much the passenger paid for the ticker.
- Cabin -- Which cabin the passenger was in.
- Embarked -- Where the passenger boarded the Titanic.

This can be useful to download from Kaggle.
```
pip install kaggle
# Download an API key from https://www.kaggle.com/<username>/account 
# Save kaggle.json in the folder .kaggle
kaggle competitions download -c titanic
```

### Tasks

1. Create a function according to your team and number.

2. Test your function with your colleague, explain him your solution, and ask him some suggestions. What could be improved?

3. Improve your function, use the feedback you get from your colleague.

4. A team A has to join a team B. Participant 1 will create a repository with his function and students 2, 3 and 4 have to pull the repo, add the function and push. 

5. Participant 3 will review the python project and make sure that all functions are working properly. 

5. Participant 4 will explain results, if possible submit predictions in Kaggle.

### Definitions

**Team A**

**Participant 1**

Create a function that reads the Titanic dataset _read_data_ and one that remove useless variables, create new variables,
 imput nulls values _clean_data_ and save in a file _cleaning.py_.

**Participant 2**

Create a function _visualization_ to plot categorical variables given a dataset, a variable, and a target. 
Save in a file with the function's name. The plot has to indicate the relation input variable vs. label.


**Team B**

**Participant 3**

Create a function _trainmodel_ that returns predicted values for training and test sets and save the model for posterior use. The function needs the inputs: target variable, type of model an a list of features. 
Save your code in _modelling.py_. It would be nice if you create also a function _select_model_  to select a model.

**Participant 4**

Create a function _eval_model_ that evaluates a model with proper metrics in a file _evaluation.py_. The function has as inputs the predicted values and labels from test set.

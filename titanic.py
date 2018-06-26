from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from src.cleaning import read_data, clean_data
from src.visualization import plot_variables
from src.modelling import select_model, train_model
# from src.evaluation import eval_model

# 1. Cleaning
# Reading data
training, test = read_data("data", ["train.csv", "test.csv"], ["Sex", "Pclass", "Cabin"])

# Visualization
# plot_variables(training, "Cabin", "Age")

# training.groupby('Survived').agg('count')
# test.head()

# Cleaning data
training, encoders_dic = clean_data(training, na_action=True)
test, _ = clean_data(test, na_action=True, test_set=True, encoders=encoders_dic)

# 2. Modelling

models = list()
models.append(('LR', LogisticRegression(class_weight='balanced')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NN', MLPClassifier()))
models.append(('CART', DecisionTreeClassifier(class_weight='balanced')))
models.append(('RF', RandomForestClassifier(class_weight='balanced')))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(class_weight='balanced')))
models.append(('GBC', GradientBoostingClassifier()))

seed = 300
scoring = 'recall'
variables = ["Age", "Fare", "SibSp", "Parch", "Pclass", "Sex", "Title"]

results = select_model(training, "Survived", variables, models, scoring, seed)

model = SVC()
predictions = train_model(training, test, "Survived", variables, model)

# 3. Evaluation
#eval_model(y_test, predictions)


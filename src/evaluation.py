from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


def eval_model(y_test, predictions):
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(roc_auc_score(y_test, predictions))

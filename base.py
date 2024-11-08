
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def splitter(data, y_var='DISEASE'):

   # Splitting the data into dependent & independent variables -
    X = data.drop(columns=y_var, axis=1).values
    y = data[y_var].values

    return X, y


from sklearn.preprocessing import StandardScaler

def standardizer(X_train, X_test):
    
    # Standardizing the data -
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_scaled = np.concatenate([X_train_scaled, X_test_scaled], axis=0)
    return X_scaled, X_train_scaled, X_test_scaled


def standardize(X):
    
    # Standardizing the data -
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    return X_scaled


def model_train(model_obj, X_train, y_train, **kwargs): 

    model_obj.fit(X_train, y_train, **kwargs)
    return model_obj


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def model_eval(model_obj, X_train, X_test, y_train, y_test):

    y_pred_test = model_obj.predict(X_test)
    y_pred_test_proba = model_obj.predict_proba(X_test)[:, 1]
    
    print("Train accuracy: {:.2f}%".format(accuracy_score(y_train, model_obj.predict(X_train)) * 100))
    print("Test accuracy: {:.2f}%".format(accuracy_score(y_test, model_obj.predict(X_test)) * 100))
    print("F1 Score: {:.2f}".format(f1_score(y_test, y_pred_test)))
    print("Precision: {:.2f}".format(precision_score(y_test, y_pred_test)))
    print("Recall: {:.2f}".format(recall_score(y_test, y_pred_test)))
    print("ROC AUC Score: {:.2f}".format(roc_auc_score(y_test, y_pred_test_proba)))
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    tpr = tp/(tp+fn)
    tnr = tn/(fp+tn)
    fpr = fp/(tp+fn)
    fnr = fn/(fp+tn)
    
    print("Type 1 Error: {:.2f}".format(fpr))
    print("Type 2 Error: {:.2f}".format(fnr))
    print("Sensitivity: {:.2f}".format(tpr))
    print("Specificity: {:.2f}\n".format(1-fpr))
    
    return y_pred_test, y_pred_test_proba
    
    
def show_pred(y_pred_test, y_pred_test_proba):
    pred = pd.DataFrame({'Probability': y_pred_test_proba,
                          'Class': y_pred_test})
    print("\n", pred)

    
from sklearn.model_selection import KFold, cross_val_score
  
def cross_val(model_obj, X, y, scoring='f1'):
    kfold = KFold(n_splits=5)
    
    score = np.mean(cross_val_score(model_obj, X, y, cv=kfold, scoring=scoring, n_jobs=-1)) 
    print("Cross Validation Score: {:.2f}".format(score))
    
    
from sklearn.metrics import roc_auc_score, roc_curve

def roc_auc_curve_plot(model_obj, X_test, y_test): 
    logit_roc_auc = roc_auc_score(y_test, model_obj.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model_obj.predict_proba(X_test)[:,1])
    
    plt.figure()
    plt.plot(fpr, tpr, label='(area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


from sklearn.metrics import precision_recall_curve

def precision_recall_curve_plot(model_obj, X_test, y_test):
    y_pred_proba = model_obj.predict_proba(X_test)[:,1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    threshold_boundary = thresholds.shape[0]
    
    # plot precision
    plt.plot(thresholds, precisions[0:threshold_boundary], label='precision')
    # plot recall
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recalls')
    
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    
    plt.xlabel('Threshold Value'); plt.ylabel('Precision and Recall Value')
    plt.legend(); plt.grid()
    plt.show()



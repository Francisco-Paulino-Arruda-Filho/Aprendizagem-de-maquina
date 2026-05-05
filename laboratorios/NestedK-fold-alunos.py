#!/usr/bin/env python
# coding: utf-8

# # Nested K-fold

# ## Carrega os dados

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, auc
from itertools import product

data = np.genfromtxt("californiabin.csv", delimiter=',')
x = data[:,:-1]
y = data[:,-1].astype(int)

print(f"Número de amostras: {x.shape[0]}")
print(f"Número de dimensões: {x.shape[1]}")
print(f"Amostras por classe: {np.unique(y, return_counts=True)}")


# ## Funções principais

# In[2]:


def run_nested_cv(x, y, model_class, grid, external_kfold=5, internal_kfold=5,
                  scale_flag=True, verbose=True, random_state=12345):

    metrics = {
        'acc': [],
        'rec': [],
        'prec': [],
        'f1': [],
        'roc_auc': [],
        'roc_curve': [],
        'pr_auc': [],
        'pr_curve': [],
        'conf_mat': []
    }

    kf_outer = KFold(n_splits=external_kfold, shuffle=True, random_state=random_state)

    best_model = None

    for fold, (train_idx, test_idx) in enumerate(kf_outer.split(x)):

        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scale_flag:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        best_params = inner_loop(
            X_train, y_train,
            model_class, grid,
            internal_kfold=internal_kfold,
            scale_flag=scale_flag,
            random_state=random_state
        )

        model = model_class(**best_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # 🔹 Probabilidades (se existirem)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        # 🔹 Métricas básicas
        metrics['acc'].append(accuracy_score(y_test, y_pred))
        metrics['rec'].append(recall_score(y_test, y_pred))
        metrics['prec'].append(precision_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['conf_mat'].append(confusion_matrix(y_test, y_pred))

        # 🔹 ROC + PR (somente se houver probabilidade)
        if y_prob is not None:
            try:
                metrics['roc_auc'].append(roc_auc_score(y_test, y_prob))
            except:
                metrics['roc_auc'].append(0)

            fpr, tpr, roc_thr = roc_curve(y_test, y_prob)
            metrics['roc_curve'].append((fpr, tpr, roc_thr))

            precision, recall, pr_thr = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)

            metrics['pr_curve'].append((precision, recall, pr_thr))
            metrics['pr_auc'].append(pr_auc)

        if verbose:
            print(f"Fold {fold+1} concluído")

        best_model = model

    return metrics, best_model


# In[3]:


def inner_loop(x, y, model_class, grid, internal_kfold=5,
               scale_flag=True, verbose=True, random_state=12345):

    param_names = list(grid.keys())
    param_values = list(grid.values())

    grid_search = list(product(*param_values))

    kf_inner = KFold(n_splits=internal_kfold, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_params = None

    for values in grid_search:

        params = dict(zip(param_names, values))

        scores = []

        for train_idx, val_idx in kf_inner.split(x):

            X_train, X_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if scale_flag:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            model = model_class(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))

        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score 
            best_params = params

    return best_params


# ## Executa o grid-search para os modelos

# In[4]:


external_kfold = 10
internal_kfold = 5

methods_summary = {'LR' : {'class': LogisticRegression, 'scale': True},
                   'QDA': {'class': QuadraticDiscriminantAnalysis, 'scale': False},
                   'GNB': {'class': GaussianNB, 'scale': False},
                   'KNN': {'class': KNeighborsClassifier, 'scale': True},
                   'DT' : {'class': DecisionTreeClassifier, 'scale': False}}

# Logistic regression
methods_summary['LR']['grid'] = {'solver' : ['liblinear'],
                                'penalty': ['l1', 'l2'],                              # penalty
                                'C': 1/np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])} # C - inverse regularization

methods_summary['QDA']['grid'] = {'priors': [None]}
methods_summary['GNB']['grid'] = {'priors': [None]}

# KNN
methods_summary['KNN']['grid'] = {'n_neighbors': np.arange(1,12,2), # n_neighbors
                                  'p': [1, 1.5, 2]}                 # p - Minkowski
# Decision Tree
methods_summary['DT']['grid'] = {'criterion': ['gini', 'entropy'],                # criterion
                                 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, None]} # max_depth

trained_models = {}
for method, info in methods_summary.items():
    print(f"\n[{method}] Running nested K-fold...")
    metrics, best_model = run_nested_cv(x=x, y=y, model_class=info['class'],
                                        grid=info['grid'], scale_flag=info['scale'],
                                        external_kfold=external_kfold, internal_kfold=internal_kfold)
    trained_models[method] = {'metrics': metrics, 'model': best_model}


# Results
results = { method : {key: info['metrics'][key] for key in ['acc', 'rec', 'prec', 'f1', 'roc_auc', 'pr_auc']} for method, info in trained_models.items() }
results_curves = { method : {key: info['metrics'][key] for key in ['roc_curve', 'pr_curve']} for method, info in trained_models.items() }
results_conf_mats = { method : info['metrics']['conf_mat'] for method, info in trained_models.items() }


# ## Tabela de resultados do Nested K-fold

# In[5]:


table = pd.DataFrame(results).T
table[table.columns.difference(['roc_auc', 'pr_auc'])] = table[table.columns.difference(['roc_auc', 'pr_auc'])].map(lambda x: f"{np.mean(x):.2%} +- {1.96*np.std(x)/np.sqrt(len(x)):.2%}")
table[['roc_auc', 'pr_auc']] = table[['roc_auc', 'pr_auc']].map(lambda x: f"{np.mean(x):2.2f} +- {1.96*np.std(x)/np.sqrt(len(x)):2.2f}")
table.columns = ['Accuracy', 'Recall', 'Precision', 'F1-score', 'ROC-AUC', 'PR-AUC']
table.index = results.keys()
def extract_from_text(text):
    return float(text.split('%')[0]) if '%' in text else float(text.split('+-')[0])
table.style.apply(lambda col: [ 'font-weight:bold; color:red' if extract_from_text(x)==col.apply(extract_from_text).max() else '' for x in col ])


# ## Curvas ROC/PRC médias

# In[6]:


fig, axs = plt.subplots(1+len(results_curves), 2, figsize=(10, (1+len(results_curves)) * 5))
for i, (method, curves) in enumerate(results_curves.items()):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = results[method]['roc_auc']
    for curve in curves['roc_curve']:
        fpr, tpr, t = curve
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0) / np.sqrt(len(tprs))
    tprs_upper = np.minimum(mean_tpr + 1.96*std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 1.96*std_tpr, 0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs) / np.sqrt(len(aucs))

    axs[-1,0].plot(mean_fpr, mean_tpr, label=fr'{method} (AUC = {mean_auc:0.2f} $\pm$ {std_auc:0.2f})')
    axs[-1,0].set_xlabel("FPR")
    axs[-1,0].set_ylabel("TPR")
    axs[-1,0].set_title("Mean ROC curves")
    axs[-1,0].legend()

    axs[i,0].plot(mean_fpr, mean_tpr, color='blue', label=fr'Mean ROC (AUC = {mean_auc:0.2f} $\pm$ {std_auc:0.2f})')
    axs[i,0].fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=.2, label=r'$\pm$ 1.96 std. error')
    axs[i,0].set_xlabel("FPR")
    axs[i,0].set_ylabel("TPR")
    axs[i,0].set_title(f"ROC curves over the outer folds - {method}")
    axs[i,0].legend()

    mean_rec = np.linspace(0, 1, 100)
    prcs = []
    aucs = list(results[method]['pr_auc'])
    for curve in curves['pr_curve']:
        prc, rec, t = curve
        interp_prc = np.interp(mean_rec, rec[::-1], prc[::-1])
        prcs.append(interp_prc)
        aucs.append(auc(rec[::-1], prc[::-1]))

    mean_prc = np.mean(prcs, axis=0)
    std_prc = np.std(prcs, axis=0) / np.sqrt(len(prcs))
    prcs_upper = np.minimum(mean_prc + 1.96*std_prc, 1)
    prcs_lower = np.maximum(mean_prc - 1.96*std_prc, 0)
    mean_auc = auc(mean_rec, mean_prc)
    std_auc = np.std(aucs) / np.sqrt(len(aucs))

    axs[-1,1].plot(mean_rec, mean_prc, label=fr'{method} (AUC = {mean_auc:0.2f} $\pm$ {std_auc:0.2f})')
    axs[-1,1].set_xlabel("Recall")
    axs[-1,1].set_ylabel("Precision")
    axs[-1,1].set_title("Mean PR curves")
    axs[-1,1].legend()

    axs[i,1].plot(mean_rec, mean_prc, color='blue', label=fr'Mean PRC (AUC = {mean_auc:0.2f} $\pm$ {std_auc:0.2f})')
    axs[i,1].fill_between(mean_rec, prcs_lower, prcs_upper, color='blue', alpha=.2, label=r'$\pm$ 1.96 std. error')
    axs[i,1].set_xlabel("Recall")
    axs[i,1].set_ylabel("Precision")
    axs[i,1].set_title(f"PR curves over the outer folds - {method}")
    axs[i,1].legend()


# ## Matrizes de confusão médias

# In[7]:


for i, (method, conf_mat) in enumerate(results_conf_mats.items()):
    cm_mean = np.mean(conf_mat, axis=0) 
    cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)  
    ConfusionMatrixDisplay(cm_norm).plot(values_format=".1%")
    plt.title(f"Average confusion matrix over the outer folds - {method}")


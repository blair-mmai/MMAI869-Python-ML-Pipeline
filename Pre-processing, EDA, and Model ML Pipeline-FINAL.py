#!/usr/bin/env python
# coding: utf-8

## 1.0 This Notebook is to set up a pipeline for Preprocessing Data, fitting a model, and then evaluating the results
## Parts of this has been pipeline including dataset have been removed as part of an NDA agreement.

# This pipeline was formed based off of sample code in OReilly's Book on Machine Learning in Python.

# ## 1.1 import packages

# In[1]:


import datetime
print(datetime.datetime.now())


# In[2]:


#data analysis and processing
import pandas as pd
import numpy as np
import random as rnd
import os as system
from imblearn.under_sampling import RandomUnderSampler


# In[3]:


#data visulaization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas_profiling
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 


#machine learning supervised algorithms
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

#machine learning clustering algorithms

from sklearn.neighbors import DistanceMetric
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score 
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA 

#for finetuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


#sklearn accuracy and model evaluation
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report


# ## 1.2 Import data and analyze

# In[4]:


# Combined the customer.csv, loyalty, vehicle, insurance, service
# If it errors out make sure the path below is pointing to the correct datasets in the shared one drive folder
df_train = pd.read_csv(r".\Datasets-Original\master_final_V2.csv", encoding='latin-1')


# In[5]:


#analyze which columns are numeric and which are categorical to choose for training down the pipeline

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categoricals = ['object']

numeric_columns = df_train.select_dtypes(include=numerics)
categorical_columns = df_train.select_dtypes(include=categoricals)

num_attribs = numeric_columns.columns.values.tolist()
num_attribs.remove('disposalLoyalty_x')
cat_attribs = categorical_columns.columns.values.tolist()
cat_attribs.remove('contactid_h')

print (num_attribs)

print ('**************************************************************')

print (cat_attribs)


# In[8]:


#Automatically split the numeric and categorical columns above, however use this cell if you want to manually pick certain features

labels = ['disposalLoyalty_x']

#num_attribs = ['servicetime',
 #'Xxxcommercialelectronicmessages', 'modelyear_x', 'companyindicator',
 #'customerpay_a_x', 'Amount', 'Rate', 'numberactiveXxx', 'numberhistoricXxx',
 #'odometerreading_x', 'province', 'Newlastservicedate', 'term',
 #'totalmonthlypayment_a', 'warrantypay_a_x', 'ageGroup']

#cat_attribs = ['model_x', 'series_x']


# In[9]:


# output new df based on choices above and drop na's
# ***note *** might have to have something to clean, right now dropping all nas one team should focus on cleaning the data

df_chosen_features = df_train[labels + num_attribs + cat_attribs]


# In[10]:


df_chosen_features.dropna(inplace=True)
df_chosen_features.reset_index(inplace=True)


# In[11]:


# found some nifty thing called pandas-profiling from Steve Thom's notebook. Downloaded it using conda install -c anaconda pandas-profiling
# does more than just df.describe() and df.describe(include='O')
pandas_profiling.ProfileReport(df_chosen_features, check_correlation=True)


# ## 1.3 Visualize some Variables with SNS and Matplotlib

# In[13]:


## check out distribution of non loyal to loyal with various features
#sns.set(style="ticks")
##df_for_plot = df_chosen_features[df_chosen_features['servicetimedays_y'] >= 0]
#sns.pairplot(df_chosen_features, hue="disposalLoyalty_x");


# # 2.0 Start creating a Pipeline

# ## 2.1 Build prep for pre-processing

# In[14]:


#We can create custom Transformers using scikit learn, all you need to do is import BaseEstimator, and TransformerMixin
#from sklearn.base:

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

#Below class selects the Dataframe column attributes, will be used to select the numerical and categorical columns
#So that they can be isolated and prepared seperately as they have different preperation steps

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X[self.attribute_names].values, columns=self.attribute_names)

#Below class creates encoded category features in a pandas dataframe

class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, is_series = False):
        self.is_series = is_series
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        encoder = OneHotEncoder(sparse=False)
        first_iter = True
        if self.is_series == False:
            for i in X.columns:
                if first_iter == True:
                    X_i_encoded, X_i_categories = X[i].factorize()
                    X_i_1hot = encoder.fit_transform(X_i_encoded.reshape(-1,1))
                else:
                    X_i_encoded, X_i_categories = X[i].factorize()
                    X_i_1hot = hstack([X_i_1hot, encoder.fit_transform(X_i_encoded.reshape(-1,1))]).toarray()                                      
            return X_i_1hot
        else:
            return encoder.fit_transform(X.reshape(-1,1))
        


# In[15]:


#let's put it all together by running the data preperation pipelines we built
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

#this num_pipeline applies a transformation to standard scale the data
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('std_scaler', StandardScaler()),
])

#this cat_pipeline takes categorical data and one hot encodes it
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CategoryEncoder()),
])



#this combines above pipelines in order
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    
])


# ## 2.2 Create stratify on labels and create test and train dfs and then downsample it

# In[17]:


X = df_chosen_features.drop("disposalLoyalty_x", axis=1)
y = df_chosen_features["disposalLoyalty_x"].copy()


# In[18]:


#new way is to use a class from scikit learn called stratifiedsplit which will split while keeping distribution between the train
#and test set the same accross a certain feature...in this case the Cost category of the house which is in 50K intervals

X_prepared = full_pipeline.fit_transform(X)

from sklearn.model_selection import StratifiedShuffleSplit


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df_chosen_features, df_chosen_features["disposalLoyalty_x"]):
    X_train, X_test = X_prepared[train_index], X_prepared[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # want to split the X where it is not one hot encoded and turned into a numpy array for diving in and decision trees
    X_train_tree, X_test_tree = X.iloc[train_index,1:], X.iloc[test_index,1:]


# In[19]:


rus = RandomUnderSampler(random_state=0)
X_train, y_train = rus.fit_resample(X_train, y_train)


# In[20]:


X_train.shape


# In[21]:


X.head()


# ## 2.3 Train Supervised models ( for loyalty prediction)

# ### Helper Functions to plot accuracy and evaluation metrics

# In[22]:


from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, auc

# Adopted from: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

def plot_boundaries(X_train, X_test, y_train, y_test, clf, clf_name, ax, hide_ticks=True):
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02));
    
    
    score = clf.score(X_test, y_test);

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]);
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1];

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8);

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100, cmap=cm_bright, edgecolors='k');
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap=cm_bright, edgecolors='k', alpha=0.6);

    ax.set_xlim(xx.min(), xx.max());
    ax.set_ylim(yy.min(), yy.max());
    if hide_ticks:
        ax.set_xticks(());
        ax.set_yticks(());
    else:
        ax.tick_params(axis='both', which='major', labelsize=18)
        #ax.yticks(fontsize=18);
        
    ax.set_title(clf_name, fontsize=28);
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=35, horizontalalignment='right');
    ax.grid();
    
    


def plot_roc(clf, X_test, y_test, name, ax, show_thresholds=True):
    y_pred_rf = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, y_pred_rf)

    ax.plot([0, 1], [0, 1], 'k--');
    ax.plot(fpr, tpr, label='{}, AUC={:.2f}'.format(name, auc(fpr, tpr)));
    ax.scatter(fpr, tpr);

    if show_thresholds:
        for i, th in enumerate(thr):
            ax.text(x=fpr[i], y=tpr[i], s="{:.2f}".format(th), fontsize=5, 
                     horizontalalignment='left', verticalalignment='top', color='black',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', alpha=0.1));
        
    ax.set_xlabel('False positive rate', fontsize=18);
    ax.set_ylabel('True positive rate', fontsize=18);
    ax.tick_params(axis='both', which='major', labelsize=18);
    ax.grid(True);
    ax.set_title('ROC Curve', fontsize=18)


# ### 2.3.1 Decision Tree
# #### One person to take over this pipeline and work with the hyper parameters, keep tuning and goal is to get better. This person should read up on decision trees and understand what the hyperparameters do in order to make an educated decision on how to increase accuracy. Please refer to pg 169 in Geron book

# In[23]:


# import some Decision tree specific packages
from sklearn.tree import export_graphviz


# In[24]:


# let's look at Gini way and Entropy criterion
clf_entropy = DecisionTreeClassifier(random_state=42, criterion="entropy",
                             min_samples_split=10, min_samples_leaf=10, max_depth=3, max_leaf_nodes=5)
clf_entropy.fit(X_train, y_train)

y_pred_dt = clf_entropy.predict(X_test)


# In[28]:


scores = cross_val_score(clf_entropy, X_train, y_train, cv=5, scoring='recall')
print(scores)
print(sum(scores)/len(scores))


# In[29]:


clf_gini = DecisionTreeClassifier(random_state=42, criterion="gini",
                             min_samples_split=10, min_samples_leaf=10, max_depth=3, max_leaf_nodes=5)
clf_gini.fit(X_train, y_train)


# #### Model Parameters

# In[30]:


print('**************** Tree parameters using entropy *****************************')
print(clf_entropy.tree_.node_count)
print(clf_entropy.tree_.impurity)
print(clf_entropy.tree_.children_left)
print(clf_entropy.tree_.threshold)

print('*************** Tree parameters using Gini *********************************')

print(clf_gini.tree_.node_count)
print(clf_gini.tree_.impurity)
print(clf_gini.tree_.children_left)
print(clf_gini.tree_.threshold)


# #### Model performance

# In[31]:


print ('All records where loyalty is 1: ' + str(len(y_test[y_test==1])))
print ('All records where loyalty is 0: ' + str(len(y_test[y_test==0])))


# In[32]:


feature_names = X_train_tree.columns
class_names = [str(x) for x in clf_entropy.classes_]


# In[33]:


print(classification_report(y_test, y_pred_dt, target_names=class_names))


# In[34]:


print(classification_report(y_test, clf_gini.predict(X_test), target_names=class_names))


# In[35]:



from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss
print('**************** Tree parameters using entropy *****************************')
print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_pred_dt)))
print("Kappa = {:.2f}".format(cohen_kappa_score(y_test, y_pred_dt)))
print("F1 Score = {:.2f}".format(f1_score(y_test, y_pred_dt)))
print("Log Loss = {:.2f}".format(log_loss(y_test, y_pred_dt)))
print('**************** Tree parameters using gini ********************************')
print("Accuracy = {:.2f}".format(accuracy_score(y_test, clf_gini.predict(X_test))))
print("Kappa = {:.2f}".format(cohen_kappa_score(y_test, clf_gini.predict(X_test))))
print("F1 Score = {:.2f}".format(f1_score(y_test, clf_gini.predict(X_test))))
print("Log Loss = {:.2f}".format(log_loss(y_test, clf_gini.predict(X_test))))


# In[58]:


# output a image of tree
#dotfile = open("dtree.dot", 'w')
#export_graphviz(clf_gini, out_file = dotfile, feature_names = (num_attribs + cat_attribs))
#dotfile.close()


# ### 2.3.2 SVM

# In[36]:


clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)


# In[37]:


# looks like it is using class imbalance to show accurate results
confusion_matrix(y_test, clf.predict(X_test))


# ### 2.3.3 Random Forests/ensemble
# 
# #### One person to take over this pipeline and work with the hyper parameters, keep tuning and goal is to get better. This person should read up on decision trees and understand what the hyperparameters do in order to make an educated decision on how to increase accuracy. Please refer to pg 191 in Geron book

# In[38]:


rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)


# In[39]:


rnd_clf.fit(X_train, y_train)


# In[40]:


# looks like it is using class imbalance to show accurate results
confusion_matrix(y_test, rnd_clf.predict(X_test))


# ### 2.3.4 KNN (K nearest neighbors)

# In[41]:


neigh = KNeighborsClassifier(n_neighbors=3)


# In[42]:


neigh.fit(X_train, y_train)


# In[56]:


scores = cross_val_score(neigh, X_train, y_train, cv=5)
f1_scores = cross_val_score(neigh, X_train, y_train, cv=5, scoring='roc_auc')
print(f1_scores)


# In[43]:


# looks like it is using class imbalance to show accurate results
confusion_matrix(y_test, neigh.predict(X_test))


# ### 2.3.5 Logistic Regression (for SNEHA or NAYEF)

# In[44]:


from sklearn.linear_model import LogisticRegression

log_reg_clf = LogisticRegression(random_state=0, solver='lbfgs')


# In[45]:


log_reg_clf.fit(X_train, y_train)


# In[55]:


scores = cross_val_score(log_reg_clf, X_train, y_train, cv=5)
f1_scores = cross_val_score(log_reg_clf, X_train, y_train, cv=5, scoring='roc_auc')
print(f1_scores)


# In[133]:


# looks like it is using class imbalance to show accurate results
confusion_matrix(y_test, log_reg_clf.predict(X_test), labels=[0,1])


# In[137]:


len(X_train_tree.columns.values[:18])


# In[128]:


log_reg_clf.coef_[:,:18][0]


# In[129]:


for features, betas in zip(X_train_tree.columns.values[:18], log_reg_clf.coef_[:,:18][0]):
    print(str(features) + '--->'+ str(betas))


# In[145]:


# positive towards loyal, negative towards disloyal
index = np.arange(len(X_train_tree.columns.values[:18]))
plt.bar(index, log_reg_clf.coef_[:,:18][0])
plt.xlabel('Features', fontsize=12)
plt.ylabel('Beta', fontsize=12)
plt.xticks(index, X_train_tree.columns.values[:18], fontsize=12, rotation=90)
plt.title('Feature Importance')
plt.show()


# ### 2.3.6 XG Boost ***Best score from default hyper-params, will now grid search this to fine tune it***

# In[47]:


XG_clf = GradientBoostingClassifier()


# In[48]:


XG_clf.fit(X_train, y_train)


# In[52]:


scores = cross_val_score(XG_clf, X_train, y_train, cv=5)
f1_scores = cross_val_score(XG_clf, X_train, y_train, cv=5, scoring='roc_auc')


# In[53]:


f1_scores


# In[60]:


confusion_matrix(y_test, XG_clf.predict(X_test))


# #### Grid Search

# In[ ]:


# kept running various grid searches and then chose the best parameters from analyzing the scores
# *** note this was a very iterative back and forth process...tried to do multiple grid searches to hone in on process, I feel
# *** like I can automate this more, don't think there is a libary for this! Will write pseudo code for this and see if I can code it
# *** when I have free time

# A parameter grid for XGBoost
params = {
        'max_depth': [3],
        'subsample': [0.8],
        'n_estimators': [150],
        'max_leaf_nodes': [80],
        }


# In[47]:


roc_auc_scorer = make_scorer(roc_auc_score)


# In[48]:


XG_clf.get_params().keys()


# In[49]:




grid_search = GridSearchCV(XG_clf, params, scoring=roc_auc_scorer, cv=5, return_train_score=True )


grid_search.fit(X_train, y_train)


# In[50]:




grid_search_1 = GridSearchCV(XG_clf, params, scoring=roc_auc_scorer, cv=5, return_train_score=True )


grid_search_1.fit(X_train, y_train)


# In[51]:


grid_search.best_params_


# In[ ]:



grid_search_2 = GridSearchCV(XG_clf, params, scoring=roc_auc_scorer, cv=5, return_train_score=True )


grid_search_2.fit(X_train, y_train)


# In[ ]:


# honing in on the n_estimators, did include it in first grid search where it said 150 was good, but isoloating this now


grid_search_3 = GridSearchCV(XG_clf, params, scoring=roc_auc_scorer, cv=5, return_train_score=True )


grid_search_3.fit(X_train, y_train)


# In[ ]:


# honing in on the n_estimators, did include it in first grid search where it said 150 was good, but isoloating this now


grid_search_4 = GridSearchCV(XG_clf, params, scoring=roc_auc_scorer, cv=5, return_train_score=True )


grid_search_4.fit(X_train, y_train)


# In[ ]:


# try to out put scores vs params chosen
for mean_score, params in zip(grid_search.cv_results_["mean_test_score"], grid_search.cv_results_["params"]):
    print(mean_score, params)


# In[ ]:


# try to out put scores vs params chosen
for mean_score, params in zip(grid_search_1.cv_results_["mean_test_score"], grid_search_1.cv_results_["params"]):
    print(mean_score, params)


# In[ ]:


# try to out put scores vs params chosen
for mean_score, params in zip(grid_search_2.cv_results_["mean_test_score"], grid_search_2.cv_results_["params"]):
    print(mean_score, params)


# In[ ]:


# try to out put scores vs params chosen, will keep it at 150, negligble diff
for mean_score, params in zip(grid_search_3.cv_results_["mean_test_score"], grid_search_3.cv_results_["params"]):
    print(mean_score, params)


# In[ ]:


# try to out put scores vs params chosen, will keep it at 150, negligble diff
for mean_score, params in zip(grid_search_4.cv_results_["mean_test_score"], grid_search_4.cv_results_["params"]):
    print(mean_score, params)


# #### Grid Searched all over and picked some parameters will evaluate fine tuned model to default

# In[61]:


XG_clf_finetuned = GradientBoostingClassifier(max_depth=3, subsample=0.8,n_estimators=150, max_leaf_nodes=80)


# In[ ]:


scores = cross_val_score(XG_clf, X_train, y_train, cv=5)
roc_auc = cross_val_score(XG_clf, X_train, y_train, cv=5, scoring='roc_auc')


scores_finetuned = cross_val_score(XG_clf_finetuned, X_train, y_train, cv=5)
roc_auc_finetuned = cross_val_score(XG_clf_finetuned, X_train, y_train, cv=5, scoring='roc_auc')


# In[ ]:


print(sum(f1_scores)/5)

print(sum(f1_scores_finetuned)/5) ## fine tuned is negligibly better


# In[62]:


XG_clf_finetuned.fit(X_train, y_train)


# In[ ]:


XG_clf_finetuned_onlynumeric


# In[63]:


confusion_matrix(y_test, XG_clf_finetuned.predict(X_test))


# #### let's pull out the best params

# In[ ]:


X_train_tree.columns[:20]


# In[ ]:


XG_clf_finetuned.feature_importances_[:20]


# In[ ]:


for feature_names, feature_importances in zip(X_train_tree.columns[:20], XG_clf_finetuned.feature_importances_[:20]):
    print(str(feature_names) + '--> ' + str(feature_importances))
    
    
# odometer, modelyear_x, and NewLastservicedate are the highest in that order


# In[ ]:


from matplotlib import pyplot
pyplot.bar(range(len(XG_clf_finetuned.feature_importances_)), XG_clf_finetuned.feature_importances_)
pyplot.show(), print(X_train.item)


# ### 2.3.7 NN

# In[96]:


NN_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 20), random_state=1)


# In[97]:


NN_clf.fit(X_train, y_train)


# In[99]:


confusion_matrix(y_test, NN_clf.predict(X_test))


# ## 2.4 Visualize all your classification models

# #### Save all models into a file so you don't have to run all the stuff from above again (note pls do this as it takes a while to train some of these models

# import pickle

# dt_model = 'dt_model.sav'
# rf_model = 'rf_model.sav'
# xg_model = 'xg_model.sav'
# nn_model = 'nn_model.sav'
# knn_model = 'knn_model.sav'
# svm_model = 'svm_model.sav'
# log_model = 'log_model.sav'
# 
# 
# pickle.dump(clf_entropy, open(dt_model, 'wb'))
# pickle.dump(rnd_clf, open(rf_model, 'wb'))
# pickle.dump(XG_clf_finetuned, open(xg_model, 'wb'))
# pickle.dump(NN_clf, open(nn_model, 'wb'))
# pickle.dump(neigh, open(knn_model, 'wb'))
# pickle.dump(clf, open(svm_model, 'wb'))
# pickle.dump(log_reg_clf, open(log_model, 'wb'))
# 
# 

# #### load a model in...if starting this notebook from scratch just load pre trained models to visualise

# In[64]:


#insert the trained classifier from above in here
fitted_classifier_for_visualization = XG_clf_finetuned


# In[65]:


# seems to be predicting non loyal pretty well, however loyal is kind of hit or miss
from yellowbrick.classifier import ClassPredictionError

visualizer_entropy = ClassPredictionError(fitted_classifier_for_visualization, classes=class_names)


visualizer_entropy.fit(X_train, y_train)
visualizer_entropy.score(X_test, y_test)
g = visualizer_entropy.poof()


# #### To get the visualization of ROC and AUC curves plug in the CLF object from Section 2.3 to visualize these curves for the specific model that was trained

# In[66]:


from yellowbrick.classifier import ROCAUC

visualizer_entropy = ROCAUC(fitted_classifier_for_visualization, classes=class_names)

visualizer_entropy.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer_entropy.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer_entropy.poof()             # Draw/show/poof the data


# In[ ]:





# In[ ]:


plt.style.use('default');
figure = plt.figure(figsize=(10, 6));    
ax = plt.subplot(1, 1, 1);
plot_roc(fitted_classifier_for_visualization, X_test, y_test, "XG boost fine tuned", ax)
plt.legend(loc='lower right', fontsize=18);
plt.tight_layout();
#plt.savefig('default-dt-roc.png'); save it as a png file


# In[ ]:


# let's look at the confusion matrix
# still kinda crappy but auc 0.74 is "good" not "great"

confusion_matrix(y_test, fitted_classifier_for_visualization.predict(X_test))


# # 2.5 Feature importance for Strategies

# In[ ]:



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions
from mlxtend.feature_selection import SequentialFeatureSelector as SFS






sfs1 = SFS(XG_clf_finetuned_onlynumeric,
          k_features= 10,
          forward=True,
          floating=False,
          verbose=2,
          scoring='recall',
          cv=5)
sfs1 = sfs1.fit(X_train_res, y_train_res)


# # 2.6 Clustering for Customer Segmentation

# In[68]:


df_chosen_features_clustering = df_train[labels + num_attribs]


df_chosen_features_clustering.dropna(inplace=True)
df_chosen_features_clustering.reset_index(inplace=True)


full_pipeline_clustering = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),])


# In[69]:


X_clustering = df_chosen_features_clustering.drop("disposalLoyalty_x", axis=1)
y_clustering = df_chosen_features_clustering["disposalLoyalty_x"].copy()


# In[75]:


df_chosen_features_clustering = df_chosen_features_clustering.drop("index", axis=1)


# In[82]:


df_chosen_features_clustering.head()


# In[ ]:


#new way is to use a class from scikit learn called stratifiedsplit which will split while keeping distribution between the train
#and test set the same accross a certain feature...in this case the Cost category of the house which is in 50K intervals

X_prepared_clustering = full_pipeline.fit_transform(X_clustering)

#from sklearn.model_selection import StratifiedShuffleSplit


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df_chosen_features_clustering, df_chosen_features_clustering["disposalLoyalty_x"]):
    X_train_clustering, X_test_clustering = X_prepared_clustering[train_index], X_prepared_clustering[test_index]
    y_train_clustering, y_test_clustering = y_clustering[train_index], y_clustering[test_index]


# ### 2.6.1 K-means

# In[79]:


# import some K-means specific packages

from sklearn.datasets.samples_generator import make_blobs


# In[162]:


# hyperparameters
# traversed back from elbow analysis and identified 3 clusters
cluster = 3


# In[163]:


kmeans = KMeans(n_clusters=cluster, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(df_chosen_features_clustering)


# In[164]:


df_chosen_features_clustering.columns


# In[165]:


kmeans.cluster_centers_


# In[171]:


#this is how you analyze three clusters on x axis being disposal loyalty and y being service time
plt.scatter(df_chosen_features_clustering.iloc[:,0], df_chosen_features_clustering.iloc[:,15])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,15], s=300, c=['red','purple','blue'])
plt.show()


# In[87]:


# elbow method to see elbow where inertia is the lowest to help determine the right amount of clusters
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=20, random_state=0)
    kmeans.fit(df_chosen_features_clustering)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ### 2.6.2 Db Scan

# In[ ]:


clustering = DBSCAN(eps=3, min_samples=2).fit(X_prepared_clustering)


# In[ ]:


clustering.labels_


# In[ ]:


clustering


# ### 2.6.3 Dendograms

# In[114]:


# note using categorical 1-hot encoded data, might be better to just put numerical
#df_chosen_features_clustering = pd.DataFrame(df_chosen_features_clustering)
pca = PCA(n_components = 2) 
X_principal_clustering = pca.fit_transform(df_chosen_features_clustering) 
X_principal_clustering = pd.DataFrame(X_principal_clustering) 
X_principal_clustering.columns = ['P1', 'P2']


# In[115]:


plt.figure(figsize =(8, 8)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(X_principal_clustering, method ='ward')))






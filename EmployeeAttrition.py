#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from imblearn.over_sampling  import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB,GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, confusion_matrix, f1_score, roc_curve,auc
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.simplefilter('ignore')


# In[3]:


ca=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition (1).csv")


# In[3]:


ca.head()


# In[4]:


ca.info()


# In[5]:


from pandas_profiling import ProfileReport
ProfileReport(ca)


# In[6]:


ca.duplicated().sum()


# In[4]:


ca.drop_duplicates(inplace=True)


# In[8]:


ca.isnull().sum()


# Based on the Profiling Report, we can drop EmployeeCount, StandardHours and Over18 columns

# In[5]:


ca.drop(["EmployeeCount","StandardHours","Over18"],axis=1,inplace=True)


# In[10]:


ca['Attrition'].value_counts()


# In[11]:


attr_yes = len(ca[ca['Attrition'] == 'Yes'])
attr_no = len(ca[ca['Attrition'] == 'No'])
class_distribution_ratio = attr_no/attr_yes
class_distribution_ratio 


# # EDA for Categorical Variables

# In[12]:


plt.figure(figsize=(10,6))
ax=sns.countplot(data=ca,x='Gender',hue='Attrition')

for p in ax.patches:
   ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.03))


# We can observe that, attrition ratio of both the male and female employees is nearly equal.
# 

# In[6]:


ca['Gender']=ca['Gender'].map({'Female':0,'Male':1})


# In[14]:


ca['BusinessTravel'].unique()


# In[15]:


travel_attr=ca.groupby(by=['BusinessTravel','Attrition']).count()['EmployeeNumber'].unstack()


# In[16]:


travel_attr


# In[17]:


travel_attr['Yes']/(travel_attr['No']+travel_attr['Yes'])


# The above data shows that it is more comnmon for employees to leave the company when they are given the opportunity to 'Travel_Frequently' as compared to employees who do not travel much.

# In[7]:


ca['BusinessTravel']=ca['BusinessTravel'].map({'Travel_Rarely':0,'Travel_Frequently':1,'Non-Travel':2})
ca.head()


# In[19]:


ca['Department'].unique()


# In[20]:


plt.figure(figsize=(10,6))
sns.countplot(data=ca,x='Department',hue='Attrition')


# In[21]:


dept_attr=ca.groupby(by=['Department','Attrition']).count()['EmployeeNumber'].unstack()


# In[22]:


dept_attr


# In[23]:


dept_attr['Yes']/(dept_attr['No']+dept_attr['Yes'])


# The attrition rate is significantly more in the Sales department and HR departments.

# In[24]:


sales_emp=ca[(ca.Department == 'Sales') & (ca.Attrition == 'Yes') ]


# In[25]:


sales_emp['JobRole'].unique()


# In[26]:


sales_job_roles=sales_emp.JobRole.value_counts().values
sales_job_names=sales_emp.JobRole.value_counts().index


# In[27]:


plt.pie(sales_job_roles,labels= sales_job_names,autopct="%1.2f%%")


# In the Sales department Sales Executives have the highest attrition rate

# In[8]:


ca['Department']=ca['Department'].map({'Sales':0,'Human Resources':1,'Research & Development':2})
ca.head()


# In[29]:


ca['Education'].unique()


# In[15]:


ca.drop("Education",axis=1,inplace=True)


# In[31]:


ca['EducationField'].unique()


# In[32]:


plt.figure(figsize=(10,6))
ax=sns.countplot(data=ca,x='EducationField',hue='Attrition')

for p in ax.patches:
   ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.03))
               
plt.show()


# In[33]:


ed_attr=ca.groupby(by=['EducationField','Attrition']).count()['EmployeeNumber'].unstack()


# In[34]:


ed_attr


# In[35]:


ed_attr['Yes']/(ed_attr['No']+ed_attr['Yes'])


# From the above plot and data we can infer that employees with a Technical Degree and Human Resources have higher attrition rate.

# In[36]:


tech_degree=ca[(ca.EducationField == 'Technical Degree')]


# In[37]:


jobroles=tech_degree.JobRole.value_counts()


# In[38]:


print(jobroles)


# In[39]:


plt.figure(figsize=(20,6))
sns.countplot(data=tech_degree,x='JobRole')


# Employees with a Technical Degree are more likely to work as Research Scientists
# 

# In[40]:


ca['EducationField'].unique()


# In[9]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
ca['EducationField']=LE.fit_transform(ca.EducationField)


# In[42]:


ca['DistanceFromHome'].unique()


# In[10]:


ca['DistanceFromHome'] = pd.cut(ca['DistanceFromHome'], bins=[0, 5, 10, 15, 20, 25, 30])


# In[44]:


ca.head()


# In[11]:


plt.figure(figsize=(20,10))

ax=sns.countplot(data=ca,x='DistanceFromHome',hue='Attrition')

for p in ax.patches:
   ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.03))


# From the above plots, we can infer that although the employees prefer to stay near the vicinity of the office,  DistanceFromHome isn't really affecting the attrition rate of the company. 

# In[11]:


ca.drop("DistanceFromHome",axis=1,inplace=True)


# In[47]:


ca['JobLevel'].unique()


# In[48]:


plt.figure(figsize=(10,6))
sns.countplot(data=ca[ca.Attrition=="Yes"],x='JobLevel')


# Hence employees at lower Job levels are more likely to leave.

# In[49]:


ca['JobRole'].unique()


# In[12]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
ca['JobRole']=LE.fit_transform(ca.JobRole)


# In[51]:


ca['MaritalStatus'].unique()


# In[52]:


plt.figure(figsize=(10,6))
sns.countplot(data=ca,x='MaritalStatus',hue='Attrition')


# Single people are more likely to leave the company when compared to Married or Divorced people.

# In[13]:


ca['MaritalStatus']=ca['MaritalStatus'].map({'Single':0,'Married':1,'Divorced':2})


# In[54]:


ca['NumCompaniesWorked'].unique()


# In[55]:


com_plot = ca.groupby(['NumCompaniesWorked', 'Attrition']).size().reset_index().pivot(columns='Attrition', index='NumCompaniesWorked',values=0)
com_plot= com_plot[['Yes','No']]


# In[56]:


com_plot.plot(kind='bar', stacked=True, figsize=(15, 10))


# From the above data we can infer that employees who worked in 1 company are more likely to leave.

# In[57]:


ca['OverTime'].unique()


# In[58]:


plt.figure(figsize=(10,6))
sns.countplot(data=ca,x='OverTime',hue='Attrition')


# In[59]:


ot_attr=ca.groupby(by=['OverTime','Attrition']).count()['EmployeeNumber'].unstack()


# In[60]:


ot_attr['Yes']/(ot_attr['No']+ot_attr['Yes'])


# Employees who are working OverTime are likely to leave thye company

# In[14]:


ca['OverTime']=ca['OverTime'].map({'Yes':1,'No':0})


# In[62]:


ca['PerformanceRating'].unique()


# In[63]:


plt.figure(figsize=(10,6))
sns.countplot(data=ca,x='PerformanceRating',hue='Attrition')


# Employees with a lower PerformanceRating of 3 are more likely to leave the company.

# In[64]:


ca['StockOptionLevel'].nunique()


# In[65]:


plt.figure(figsize=(10,6))
sns.countplot(data=ca,x='StockOptionLevel',hue='Attrition')


# Employees who haven't invested in any of the company stocks have a higher skope of attrition.

# In[15]:


ca['StockOptionLevel'] = ca['StockOptionLevel'].apply(lambda x:1 if x>0 else 0)


# In[67]:


ca['StockOptionLevel'].unique()


# In[68]:


ca['TrainingTimesLastYear'].unique()


# In[69]:


tr_plot = ca.groupby(['TrainingTimesLastYear', 'Attrition']).size().reset_index().pivot(columns='Attrition', index='TrainingTimesLastYear',values=0)
tr_plot= tr_plot[['Yes','No']]


# In[70]:


tr_plot.plot(kind='bar', stacked=True, figsize=(15, 10))


# Number of Employees who had trainings 2-3 times last year are more and they are likely to leave

# # EDA for Numeric variables

# In[23]:


attr=ca[ca['Attrition']=="Yes"]
noattr=ca[ca['Attrition']=="No"]

def numerical_column_viz(col_name):
    f,ax = plt.subplots(1,2, figsize=(18,6))
    sns.kdeplot(attr[col_name],hue=ca['Attrition'],shade=True,ax=ax[0])
    sns.kdeplot(noattr[col_name],legend=True, ax=ax[0], shade=True, color='salmon')
    
    sns.boxplot(y=col_name, x='Attrition',data=ca, palette='Set3', ax=ax[1])
    plt.show()


# In[72]:


numerical_column_viz("Age")


# In[73]:


sns.jointplot(data=ca,x='Age',y='TotalWorkingYears',hue='Attrition',height=10)


# In[74]:


numerical_column_viz("DailyRate")


# In[75]:


numerical_column_viz("HourlyRate")


# In[76]:


numerical_column_viz("MonthlyRate")


# Retaining just MonthlyRate feature since the trends for MonthlyRate, DailyRate, HourlyRate are similar

# In[16]:


ca.drop(["DailyRate","HourlyRate"],axis=1,inplace=True)


# In[78]:


numerical_column_viz("MonthlyIncome")


# In[80]:


numerical_column_viz("PercentSalaryHike")


# In[81]:


numerical_column_viz("YearsAtCompany")


# In[82]:


sns.jointplot(data=ca,x='Age',y='YearsAtCompany',hue='Attrition',height=10)


# In[83]:


numerical_column_viz("YearsInCurrentRole")


# In[84]:


numerical_column_viz("YearsSinceLastPromotion")


# In[85]:


numerical_column_viz("YearsWithCurrManager")


# From all of the above plots we can infer that attrition is generally more in employees with less than 2 years of experience.

# # EDA for Features with Ratings

# In[17]:


# 'EnviornmentSatisfaction', 'JobInvolvement', 'JobSatisfacction', 'RelationshipSatisfaction', 'WorklifeBalance' can be clubbed into a single feature 'TotalSatisfaction'

ca['Total_Satisfaction'] = (ca['EnvironmentSatisfaction'] + 
                            ca['JobInvolvement'] + 
                            ca['JobSatisfaction'] + 
                            ca['RelationshipSatisfaction'] +
                            ca['WorkLifeBalance']) /5 

# Drop Columns
ca.drop(['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'], axis=1, inplace=True)


# In[87]:


f,ax = plt.subplots(1,2, figsize=(18,6))
sns.countplot(data=ca,x='Total_Satisfaction',hue='Attrition',ax=ax[0])
sns.boxplot(y='Total_Satisfaction', x='Attrition',data=ca, palette='Set3',ax=ax[1])


# The Attrition is more when the Total_satisfaction ratings of the employees is between 2.2 to 3

# In[18]:


ca.drop("EmployeeNumber",axis=1,inplace=True)


# In[89]:


k=1
plt.figure(figsize=(40, 40))
for col in ca:
  if col=="Attrition":
    continue
  yes = ca[ca['Attrition'] == 'Yes'][col]
  no = ca[ca['Attrition'] == 'No'][col]
  plt.subplot(5, 5, k)
  plt.hist(yes, bins=25, alpha=0.5, label='yes', color='b')
  plt.hist(no, bins=25, alpha=0.5, label='no', color='r')
  plt.legend(loc='upper right')
  plt.title(col)
  k+=1


# In[90]:


#Plotting a Heatmap to identify the correlation between the independant features and the dependant feature
corr = ca.corr()
f, ax = plt.subplots(figsize=(20, 10))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr, annot=True, cmap=cmap)


# Checking for Multicollinearity from the HeatMap and Removing highly correlated features

# In[19]:


ca.drop(["JobLevel","TotalWorkingYears","JobRole","YearsAtCompany"],axis=1,inplace=True)


# In[20]:


ca['Attrition']=ca['Attrition'].map({'Yes':1,'No':0})


# In[94]:


ca.info()


# In[21]:


y= ca['Attrition']


# In[22]:


X= ca.drop(columns='Attrition')


# In[393]:


over = SMOTE(sampling_strategy=0.4)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under), ('model',MLPClassifier(random_state=0,max_iter=10000))]
pipeline = Pipeline(steps=steps)


# In[394]:


pipeline


# In[97]:


X.shape


# In[23]:


X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)


# In[99]:


print(f"Train data shape: {X_train.shape}, Test Data Shape {X_test.shape}")


# Standardization

# In[24]:


#Standardization to make mean=0, sd=1

standard_scaler = StandardScaler()

X_train_standardized = standard_scaler.fit_transform(X_train)
X_test_standardized = standard_scaler.transform(X_test)

X_standardized = standard_scaler.fit_transform(X)


# In[25]:


##Normalization to make values between 0 and 1

X_train_normalized = min_max_scaler.fit_transform(X_train)
X_test_normalized = min_max_scaler.transform(X_test)

X_normalized = min_max_scaler.fit_transform(X)


# In[262]:


def tune_hyperparameters_lr(model,X,y):
    param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"],"solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
    grid_search = GridSearchCV(model,param_grid=param_grid)
    grid_search.fit(X,y)
    print("Best Params: ",grid_search.best_params_)


# In[33]:


def tune_hyperparameters_svc(model,X,y):
    param_grid = {'C' : [0.1,1,10,100,1000],'kernel' : ['linear','rbf','sigmoid']}
    grid_search = GridSearchCV(model,param_grid=param_grid)
    grid_search.fit(X,y)
    print("Best Params: ",grid_search.best_params_)


# In[253]:


def tune_hyperparameters_knn(model,X,y):
  param_grid = {'n_neighbors' : np.arange(5,20,2),'leaf_size' : np.arange(1,50,5),'weights' : ['uniform','distance']}
  grid_search = GridSearchCV(model,param_grid=param_grid)
  grid_search.fit(X,y)
  print("Best Params: ",grid_search.best_params_)


# In[167]:


def tune_hyperparameters_dt(model,X,y):
  param_grid = {'criterion':['entropy','gini'],'max_depth':np.arange(1,15),'max_leaf_nodes':[5, 10, 20, 50, 100],'random_state':[0]}
  grid_search = GridSearchCV(model,param_grid=param_grid)
  grid_search.fit(X,y)
  print("Best Params: ",grid_search.best_params_)


# In[197]:


def tune_hyperparameters_rf(model,X,y):
  param_grid = {'n_estimators':[int(x) for x in np.linspace(start = 10, stop = 100, num = 5)],'max_features': ['auto', 'sqrt'],'max_depth' :[int(x) for x in np.linspace(10, 150, num = 10)],'min_samples_split':[2, 5, 10],'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]}
  grid_search = GridSearchCV(model,param_grid=param_grid)
  grid_search.fit(X,y)
  print("Best Params: ",grid_search.best_params_)


# In[217]:


def tune_hyperparameters_ada(model,X,y):
  param_grid = {'n_estimators' : [10,50,250,1000],'learning_rate' : [0.001,0.01,0.1,1.0,10]}
  grid_search = GridSearchCV(model,param_grid=param_grid)
  grid_search.fit(X,y)
  print("Best Params: ",grid_search.best_params_)


# In[240]:


def tune_hyperparameters_mlp(model,X,y):
  param_grid = {
      'activation' : ['identity','logistic','tanh','relu'],
      'solver': ['lbfgs','sgd','adam'],
      'alpha': [0.0001,0.05]
  }
  grid_search = GridSearchCV(model,param_grid=param_grid)
  grid_search.fit(X,y)
  print("Best Params: ",grid_search.best_params_)


# In[263]:


tune_hyperparameters_lr(LogisticRegression(max_iter=100000),X_train_standardized,y_train)


# In[104]:


tune_hyperparameters_lr(LogisticRegression(max_iter=100000),X_train_normalized,y_train)


# In[29]:


tune_hyperparameters_svc(SVC(random_state=0),X_train_standardized,y_train)


# In[123]:


tune_hyperparameters_svc(SVC(random_state=0),X_train_normalized,y_train)


# In[254]:


tune_hyperparameters_knn(KNeighborsClassifier(),X_train_standardized,y_train)


# In[255]:


tune_hyperparameters_knn(KNeighborsClassifier(),X_train_normalized,y_train)


# In[168]:


tune_hyperparameters_dt(DecisionTreeClassifier(),X_train_standardized,y_train)


# In[169]:


tune_hyperparameters_dt(DecisionTreeClassifier(),X_train_normalized,y_train)


# In[198]:


tune_hyperparameters_rf(RandomForestClassifier(),X_train_standardized,y_train)


# In[199]:


tune_hyperparameters_rf(RandomForestClassifier(),X_train_normalized,y_train)


# In[226]:


dt_params= {'criterion': 'gini', 'max_depth': 3, 'max_leaf_nodes': 10, 'random_state': 0}


# In[408]:


lr_params= {'C': 1.0, 'penalty': 'l1', 'solver': 'saga'}


# In[415]:


tune_hyperparameters_ada(AdaBoostClassifier(random_state=0,base_estimator=DecisionTreeClassifier(**dt_params)),X_train_standardized,y_train)


# In[416]:


tune_hyperparameters_ada(AdaBoostClassifier(random_state=0,base_estimator=DecisionTreeClassifier(**dt_params)),X_train_normalized,y_train)


# In[242]:


tune_hyperparameters_mlp(MLPClassifier(random_state=0,max_iter=10000),X_train_standardized,y_train)


# In[243]:


tune_hyperparameters_mlp(MLPClassifier(random_state=0,max_iter=10000),X_train_normalized,y_train)


# In[26]:


def train_predict_evaluate(model,X_train,y_train,X_test):
  #model=pipeline
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)

  print("Accuracy: ",accuracy_score(y_test,y_pred))
  print("Precision: ",precision_score(y_test,y_pred))
  print("Recall: ",recall_score(y_test,y_pred))
  print("F1 Score: ",f1_score(y_test,y_pred,average='weighted'))
  print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))

  
  fpr,tpr,thresholds = roc_curve(y_test,y_pred)
  plt.plot(fpr, tpr,color='green',label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
  plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC Curve")
  plt.legend(loc="lower right")
  plt.show()


# In[265]:


train_predict_evaluate(LogisticRegression(max_iter=100000),X_train,y_train,X_test)


# In[266]:


train_predict_evaluate(LogisticRegression(max_iter=100000),X_train_standardized,y_train,X_test_standardized)


# In[267]:


train_predict_evaluate(LogisticRegression(max_iter=100000),X_train_normalized,y_train,X_test_normalized)


# In[27]:


train_predict_evaluate(SVC(random_state=0),X_train,y_train,X_test)


# In[28]:


train_predict_evaluate(SVC(random_state=0),X_train_standardized,y_train,X_test_standardized)


# In[29]:


train_predict_evaluate(SVC(random_state=0),X_train_normalized,y_train,X_test_normalized)


# In[365]:


train_predict_evaluate(GaussianNB(),X_train,y_train,X_test)


# In[366]:


train_predict_evaluate(GaussianNB(),X_train_standardized,y_train,X_test_standardized)


# In[367]:


train_predict_evaluate(GaussianNB(),X_train_normalized,y_train,X_test_normalized)


# In[256]:


train_predict_evaluate(KNeighborsClassifier(),X_train,y_train,X_test)


# In[257]:


train_predict_evaluate(KNeighborsClassifier(),X_train_standardized,y_train,X_test_standardized)


# In[258]:


train_predict_evaluate(KNeighborsClassifier(),X_train_normalized,y_train,X_test_normalized)


# In[350]:


train_predict_evaluate(DecisionTreeClassifier(),X_train,y_train,X_test)


# In[351]:


train_predict_evaluate(DecisionTreeClassifier(),X_train_standardized,y_train,X_test_standardized)


# In[352]:


train_predict_evaluate(DecisionTreeClassifier(),X_train_normalized,y_train,X_test_normalized)


# In[380]:


train_predict_evaluate(RandomForestClassifier(),X_train,y_train,X_test)


# In[381]:


train_predict_evaluate(RandomForestClassifier(),X_train_standardized,y_train,X_test_standardized)


# In[382]:


train_predict_evaluate(RandomForestClassifier(),X_train_normalized,y_train,X_test_normalized)


# In[418]:


train_predict_evaluate(AdaBoostClassifier(random_state=0,base_estimator=DecisionTreeClassifier(**dt_params)),X_train,y_train,X_test)


# In[419]:


train_predict_evaluate(AdaBoostClassifier(random_state=0,base_estimator=DecisionTreeClassifier(**dt_params)),X_train_standardized,y_train,X_test_standardized)


# In[420]:


train_predict_evaluate(AdaBoostClassifier(random_state=0,base_estimator=DecisionTreeClassifier(**dt_params)),X_train_normalized,y_train,X_test_normalized)


# In[396]:


train_predict_evaluate(MLPClassifier(random_state=0,max_iter=10000),X_train,y_train,X_test)


# In[397]:


train_predict_evaluate(MLPClassifier(random_state=0,max_iter=10000),X_train_standardized,y_train,X_test_standardized)


# In[398]:


train_predict_evaluate(MLPClassifier(random_state=0,max_iter=10000),X_train_normalized,y_train,X_test_normalized)


# In[353]:


pipeline


# In[36]:


def cross_validation(model,X,y):
  #model=pipeline
  scores = cross_validate(model,X, y, cv=5,scoring=('accuracy','precision','recall','f1_weighted','roc_auc'))
  metrics = []
  metrics.append(np.mean(scores['test_accuracy']))
  metrics.append(np.mean(scores['test_precision']))
  metrics.append(np.mean(scores['test_recall']))
  metrics.append(np.mean(scores['test_f1_weighted']))
  metrics.append(np.mean(scores['test_roc_auc']))

  print("Accuracy: ",metrics[0])
  print("Precision: ",metrics[1])
  print("Recall: ",metrics[2])
  print("F1 Score: ",metrics[3])
  print("ROC_AUC Score: ",metrics[4])

  return metrics


# In[37]:


metrics = []


# In[336]:


metrics


# In[422]:


models=['Logistic Regression','SVM Classifier','Naive Bayes Classifier','Decision Tree Classifier','Random Forest Classifier','AdaBoost Classifier','Multi Layer Perceptron']


# In[338]:


metrics.append(cross_validation(LogisticRegression(max_iter=100000),X,y))


# In[339]:


metrics.append(cross_validation(LogisticRegression(max_iter=100000,C=0.1,penalty='l2',solver='newton-cg'),X_standardized,y))


# In[340]:


metrics.append(cross_validation(LogisticRegression(max_iter=100000,C=1.0,penalty='l2',solver='saga'),X_normalized,y))


# In[341]:


metrics.append(cross_validation(SVC(random_state=0),X,y))


# In[38]:


metrics.append(cross_validation(SVC(random_state=0,C=10,kernel='linear'),X_standardized,y))


# In[343]:


metrics.append(cross_validation(SVC(random_state=0,C=100,kernel='linear'),X_normalized,y))


# In[370]:


metrics.append(cross_validation(GaussianNB(),X,y))


# In[371]:


metrics.append(cross_validation(GaussianNB(),X_standardized,y))


# In[372]:


metrics.append(cross_validation(GaussianNB(),X_normalized,y))


# In[86]:


metrics.append(cross_validation(KNeighborsClassifier(),X,y))


# In[87]:


metrics.append(cross_validation(KNeighborsClassifier(leaf_size=1,n_neighbors=7,weights='uniform'),X_standardized,y))


# In[88]:


metrics.append(cross_validation(KNeighborsClassifier(leaf_size=1,n_neighbors=15,weights='distance'),X_normalized,y))


# In[356]:


metrics.append(cross_validation(DecisionTreeClassifier(),X,y))


# In[357]:


metrics.append(cross_validation(DecisionTreeClassifier(criterion='gini',max_depth= 3, max_leaf_nodes= 10, random_state= 0),X_standardized,y))


# In[358]:


metrics.append(cross_validation(DecisionTreeClassifier(criterion='gini',max_depth= 3, max_leaf_nodes= 10, random_state= 0),X_normalized,y))


# In[385]:


metrics.append(cross_validation(RandomForestClassifier(),X,y))


# In[386]:


metrics.append(cross_validation(RandomForestClassifier(bootstrap=True, max_depth=118, max_features='auto', min_samples_leaf=1,min_samples_split= 2, n_estimators=32),X_standardized,y))


# In[387]:


metrics.append(cross_validation(RandomForestClassifier(bootstrap=False,max_depth=87,max_features='auto',min_samples_leaf= 2,min_samples_split= 5,n_estimators= 10),X_normalized,y))


# In[423]:


metrics.append(cross_validation(AdaBoostClassifier(random_state=0,base_estimator=DecisionTreeClassifier(**dt_params)),X,y))


# In[424]:


metrics.append(cross_validation(AdaBoostClassifier(random_state=0,base_estimator=DecisionTreeClassifier(**dt_params),learning_rate=0.01,n_estimators=250),X_standardized,y))


# In[425]:


metrics.append(cross_validation(AdaBoostClassifier(random_state=0,base_estimator=DecisionTreeClassifier(**dt_params),learning_rate=0.01,n_estimators=10),X_normalized,y))


# In[401]:


metrics.append(cross_validation(MLPClassifier(),X,y))


# In[402]:


metrics.append(cross_validation(MLPClassifier(activation='identity',alpha=0.0001,solver='lbfgs'),X_standardized,y))


# In[403]:


metrics.append(cross_validation(MLPClassifier(activation='identity',alpha=0.0001,solver='adam'),X_normalized,y))


# In[426]:


index=[]
for x in range(len(models)):
    index.append("Without Scaling "+models[x])
    index.append("With Standardization "+models[x])
    index.append("With Normalization "+models[x])


# In[427]:


index


# In[428]:


mdf = pd.DataFrame(metrics,columns=["Accuracy","Precision","Recall","F1 Score","ROC_AUC Score"],index=index)
mdf.head(25)


# In[32]:


### Create a Pickle file using serialization 
classifier= SVC(random_state=0)


# In[45]:


import pickle
pickle_out = open("attr_predictor.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


# In[38]:


X=ca[['Age','Gender','MonthlyIncome','PerformanceRating','Total_Satisfaction']]


# In[39]:


classifier.fit(X,y)


# In[44]:


classifier.predict([[19,1,599,3,2.2]])


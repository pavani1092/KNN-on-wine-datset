
# coding: utf-8

# In[347]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url,delimiter=";")
data.head()
data.describe()


# In[348]:

#classifying the given quality as Low/High 
#replacing the given quality with low & High values
data['quality'] = np.where(data['quality']<=5 , 'Low', 'High')
data.head()


# In[349]:

#normalises the numeric values in given dataset(df) with the max and min values from trainf data
def normalize(df,trainf):
    result = df.copy()
    for feature_name in df.columns:
        if np.issubdtype(df[feature_name].dtype, np.number):
            max_value = trainf[feature_name].max()
            min_value = trainf[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[350]:

# sampling data with 75%train data and 25% test data
train_df = data.sample(frac = 0.75, random_state = 150)
test_df = data.drop(train_df.index)


# In[351]:

pd.DataFrame.hist(train_df.drop('quality',1), figsize = [15,15]);
plt.show()


# In[352]:

train = normalize(train_df, train_df) # normalise train data
test = normalize(test_df,train_df) # normalise test data with max and min values from train data
train.head()


# In[353]:

#histogram after normalising data
pd.DataFrame.hist(train.drop('quality',1), figsize = [15,15]);
plt.show()


# In[354]:

cols = set(test.columns)
cols.remove('quality')
test_cols = list(cols)
test_cols # collecting columns to compute the distance


# In[361]:

import math
from sklearn.metrics.pairwise import euclidean_distances
#calculate distance between two given rows
def calculate_distance(test_row, train_row):
    sumR = test_row-train_row
    sumR = pow(sumR,2)
    return math.sqrt(sumR.sum())
    
#calculates the distances of each record in test data with all the record in training data    
def calculate_knn(test_row, train_data):
    tData = train_data.copy()
    tData['distance'] = tData[test_cols].apply(lambda row: calculate_distance(row,test_row),axis =1)
    tData = tData.sort_values(['distance'], ascending = 1 )
    top = tData.head(35)
    res = ' '.join(top['quality'].tolist())
    return res #return a record containing top 34 values appened e.g High Low Low High... 


# In[362]:

#save the closest 35 results of each test record in result column as a string
test['result'] = test[test_cols].apply(lambda row: calculate_knn(row,train),axis =1)


# In[363]:

test['result'].head()


# In[364]:

#calculates the result of every row based in k value
def calculate_k(k,row):
    top = row.split()
    c = top[:k].count('Low')
    if c > k/2:
        return 'Low'
    return 'High'


# In[365]:

kdf = pd.DataFrame(columns=[ 'k value','accuracy' ])
# for k = 5:35 compute the accuracy to get the best k value
for i in range(5, 35, 2):
    test['result_k'] = test['result'].apply(lambda row: calculate_k(i,row))
    c = (test['result_k']==test['quality']).sum()
    c = c/len(test)
    kdf = kdf.append({'k value': i, 'accuracy': c}, ignore_index=True)

print(kdf.to_string(index=False))


# In[366]:
import seaborn as sns
sns.pointplot(x='k value', y='accuracy', data=kdf)
plt.show()


# In[367]:

kdf.loc[kdf['accuracy'].idxmax()]
kdf


# In[368]:

#computing the correlation between various features

correlation = train.corr()
sns.heatmap(correlation, annot= True, linewidths=.5)
plt.show()


# In[369]:

#eliminating the columns that are highly correlated
test_cols.remove('density')
test_cols.remove('free sulfur dioxide')
test_cols.remove('citric acid')


# In[370]:

#calculating the distances with feature reduction
test['result subset'] = test[test_cols].apply(lambda row: calculate_knn(row,train),axis =1)


# In[371]:

#calculating accuracy after feature reduction
test['result_ks'] = test['result subset'].apply(lambda row: calculate_k(15,row))
c = (test['result_ks']==test['quality']).sum()
c = c/len(test)
c


# In[372]:

#determining outliers
fig2, ax2 = plt.subplots(nrows=1, ncols=1,figsize=(20, 10))
sns.boxplot(data=train, orient="h", palette="Set2", ax = ax2)
plt.show()


# In[373]:

#calculating the interquartile region values to determine the maximum and minimum value of boxplots
q75 = train.quantile(.75)
q25 = train.quantile(.25)
iqr = q75 - q25

minc = q25 - (iqr*1.5)
maxc = q75 + (iqr*1.5)


# In[374]:

cols = set(train.columns)
cols.remove('quality')
test_cols = list(cols)
test_cols


# In[375]:

#replaces the outliers with closest min/max values
def replace_outlier(row, maxv, minv):
    if row > maxv:
        return maxv
    elif row < minv:
        return minv
    else:
        return row
    
    


# In[376]:

#transforming the training data by replacing the outliers with approximate values
train2 = train
for col in test_cols:
    train2[col] =train[col].apply(lambda row: replace_outlier(row,maxc[col], minc[col]))


# In[377]:

#plot after the removal of outliers
fig2, ax2 = plt.subplots(nrows=1, ncols=1,figsize=(20, 10))
sns.boxplot(data=train2, orient="h", palette="Set2", ax = ax2)
plt.show()


# In[378]:

#calculating the distances
test['result-outliers'] = test[test_cols].apply(lambda row: calculate_knn(row,train2),axis =1)


# In[379]:

#assigning result after approximation of outliers with final_k
test['result_o'] = test['result-outliers'].apply(lambda row: calculate_k(15,row))
c = (test['result_o']==test['quality']).sum()
c = c/len(test)
c


# In[380]:

def calulateProbability(row,k):
    top = row.split()
    c = top[:k].count('High')
    return c/k
    
    


# In[381]:

#printing result with k =15
result_df = pd.DataFrame(columns=[ 'Actual Class','Predicted Class','Posterier Probability'])
for index, row  in test.iterrows():
    result_df = result_df.append({'Actual Class': row['quality'], 'Predicted Class': row['result_o'],'Posterier Probability':calulateProbability(row['result-outliers'],15)},ignore_index=True)
result_df


# In[382]:

from sklearn.metrics import confusion_matrix
tp, fn, fp, tn = confusion_matrix(result_df['Actual Class'],result_df['Predicted Class']).ravel()


# In[383]:

#output for various values of k
def results_K(tp,fp,fn,tn):
    print("True Positive:       ",tp)
    print("False Positive:      ",fp)
    print("True Negative:       ",tn)
    print("False Negative:      ",fn)
    r = tp/(tp+fp)
    print("Precision:           ",r)
    p = tp/(tp+fn)
    print("Recall:              ",p)
    print("F-Measure:           ",2*r*p/(r+p))
    correct = tp+tn
    incorrect = fp+fn
    print("Classification rate: ",correct/(correct+incorrect))
    print("Error rate:          ",incorrect/(correct+incorrect))
    
    


# In[384]:
from sklearn import metrics
k_values = [5,11,15,29]
for k in k_values:
    res_temp = pd.DataFrame(columns=[ 'Actual Class','Predicted Class','Posterier Probability'])
    print("-----------------------------")
    print("K VALUE   =  ",k)
    for index, row  in test.iterrows():
        res_temp = res_temp.append({'Actual Class': row['quality'], 'Predicted Class': calculate_k(k,row['result-outliers']),'Posterier Probability':calulateProbability(row['result-outliers'],k)},ignore_index=True)
    tp, fn, fp, tn = confusion_matrix(res_temp['Actual Class'],res_temp['Predicted Class']).ravel()
    print("-----------------------------")
    results_K(tp,fp,fn,tn)
    print("-----------------------------")
    res_temp['Actual Class'] = np.where(res_temp['Actual Class']== 'High',1,0)
    auc = metrics.roc_auc_score(res_temp['Actual Class'],res_temp['Posterier Probability'])
    fpr, tpr, thresholds = metrics.roc_curve(res_temp['Actual Class'],res_temp['Posterier Probability'], pos_label=1)
    ax = plt.axes()
    ax.plot(fpr, tpr)
    ax.annotate('AUC: {:.2f}'.format(auc), (.8, .2))
    ax.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for K = %i' %k)
    plt.show() 


# In[385]:

cols = set(train.columns)
cols.remove('quality')
cols


# In[389]:

#off the shelf knn implementation
from sklearn.neighbors import KNeighborsClassifier

for i in range(5, 35, 2):
    # instantiate learning model (k = 3)
    knn = KNeighborsClassifier(n_neighbors=i)

    # fitting the model
    knn.fit(train[list(cols)], train['quality'])

    # predict the response
    pred = knn.predict(test[list(cols)])


    pdf = pd.DataFrame()
    pdf['quality'] = pred.tolist()
    c = (pdf['quality'].reset_index(drop=True) ==test['quality'].reset_index(drop=True)).sum()
    print("off the shelf knn accuracy: with ",i," = ", c/len(test))



# In[391]:

#off the shelf results with k=15
knn = KNeighborsClassifier(n_neighbors=15)

# fitting the model
knn.fit(train[list(cols)], train['quality'])

# predict the response
pred = knn.predict(test[list(cols)])


pdf = pd.DataFrame()
pdf['quality'] = pred.tolist()
c = (pdf['quality'].reset_index(drop=True) ==test['quality'].reset_index(drop=True)).sum()
print("off the shelf knn accuracy: with 15 = ", c/len(test))

tp, fn, fp, tn = confusion_matrix(test['quality'],pdf['quality']).ravel()
results_K(tp,fp,fn,tn)


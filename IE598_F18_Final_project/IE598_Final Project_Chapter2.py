
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("C:/Users/cloud/Downloads/MLF_GP2_EconCycle.csv")
#Exporatory Data Analysis
print(data.describe())
del data['Date']
print(data.head())
print("number of rows: ", data.shape[0], "\nnumber of columns: ", data.shape[1])

cols = [
 'T1Y Index',
 'T2Y Index',
 'T3Y Index',
 'T5Y Index',
 'T7Y Index',
 'T10Y Index',
 'CP1M',
 'CP3M',
 'CP6M',
 'CP1M_T1Y',
 'CP3M_T1Y',
 'CP6M_T1Y',
 'USPHCI',
 'PCT 3MO FWD',
 'PCT 6MO FWD',
 'PCT 9MO FWD']

cormat = data.corr()
print(cormat)
hm= pd.DataFrame(data.corr())
plt.pcolor(hm)
plt.title("Correlation Matrix")
plt.xlabel("features")
plt.ylabel("features")
plt.show()

sns.pairplot(data[cols], size=2.5)

plt.figure()
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=False,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.xlabel("features")
plt.ylabel("features")
plt.title("heatmap")
plt.show()


from sklearn.model_selection import train_test_split
X, y3, y6, y9 = data.iloc[:, :13].values, data.iloc[:, 13].values, data.iloc[:, 14].values, data.iloc[:, 15].values
X_train, X_test, y3_train, y3_test = train_test_split(X, y3, test_size=0.2,random_state=42)
X_train, X_test, y6_train, y6_test = train_test_split(X, y6, test_size=0.2,random_state=42)
X_train, X_test, y9_train, y9_test = train_test_split(X, y9, test_size=0.2,random_state=42)
##Note that because we specific the random state, X_train and X_test are the same

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.tree import DecisionTreeRegressor
tree3 = DecisionTreeRegressor()
tree6 = DecisionTreeRegressor()
tree9 = DecisionTreeRegressor()

tree3.fit(X_train_std, y3_train)
tree6.fit(X_train_std, y6_train)
tree9.fit(X_train_std, y9_train)

y3_train_pred = tree3.predict(X_train_std)
y3_test_pred = tree3.predict(X_test_std)

y6_train_pred = tree6.predict(X_train_std)
y6_test_pred = tree6.predict(X_test_std)

y9_train_pred = tree9.predict(X_train_std)
y9_test_pred = tree9.predict(X_test_std)

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
print('3 month: MSE train: %f, test: %f' % (MSE(y3_train, y3_train_pred),MSE(y3_test, y3_test_pred)))
print('3 month: R^2 train: %.3f, test: %.3f' % (r2_score(y3_train, y3_train_pred),r2_score(y3_test, y3_test_pred)))

print('6 month: MSE train: %f, test: %f' % (MSE(y6_train, y6_train_pred),MSE(y6_test, y6_test_pred)))
print('6 month: R^2 train: %.3f, test: %.3f' % (r2_score(y6_train, y6_train_pred),r2_score(y6_test, y6_test_pred)))

print('9 month: MSE train: %f, test: %f' % (MSE(y9_train, y9_train_pred),MSE(y9_test, y9_test_pred)))
print('9 month: R^2 train: %.3f, test: %.3f' % (r2_score(y9_train, y9_train_pred),r2_score(y9_test, y9_test_pred)))

def plot_residual(train, train_pred, test, test_pred):
    plt.figure()
    plt.scatter(train_pred, train_pred - train,c='steelblue', marker='o', edgecolor='white',label='Training data')
    plt.scatter(test_pred, test_pred - test,c='limegreen', marker='s', edgecolor='white',label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.title("Residual Plot")
    plt.show()

plot_residual(y3_train, y3_train_pred, y3_test, y3_test_pred)
plot_residual(y6_train, y6_train_pred, y6_test, y6_test_pred)
plot_residual(y9_train, y9_train_pred, y9_test, y9_test_pred)



from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
ridge3_scores = []
ridge6_scores = []
ridge9_scores = []

alpha_space = np.logspace(-4,0,50)

ridge3 = Ridge()
ridge6 = Ridge()
ridge9 = Ridge()

#finding the best alpha
for alpha in alpha_space:
    ridge3.alpha = alpha
    ridge3_cv_scores = cross_val_score(ridge3, X, y3, cv=10)
    ridge3_scores.append(np.mean(ridge3_cv_scores))
    
    ridge6.alpha = alpha
    ridge6_cv_scores = cross_val_score(ridge6, X, y6, cv=10)
    ridge6_scores.append(np.mean(ridge6_cv_scores))
    
    ridge9.alpha = alpha
    ridge9_cv_scores = cross_val_score(ridge9, X, y9, cv=10)
    ridge9_scores.append(np.mean(ridge9_cv_scores))

print("The best alpha value is: ", alpha_space[np.argmax(ridge3_scores)])
print("The best alpha value is: ", alpha_space[np.argmax(ridge6_scores)])
print("The best alpha value is: ", alpha_space[np.argmax(ridge9_scores)])
ridge3 = Ridge(alpha = alpha_space[np.argmax(ridge3_scores)])
ridge3.fit(X_train, y3_train)
ridge3_y_train_pred = ridge3.predict(X_train)
ridge3_y_test_pred = ridge3.predict(X_test)

ridge6 = Ridge(alpha = alpha_space[np.argmax(ridge6_scores)])
ridge6.fit(X_train, y6_train)
ridge6_y_train_pred = ridge6.predict(X_train)
ridge6_y_test_pred = ridge6.predict(X_test)

ridge9 = Ridge(alpha = alpha_space[np.argmax(ridge9_scores)])
ridge9.fit(X_train, y9_train)
ridge9_y_train_pred = ridge9.predict(X_train)
ridge9_y_test_pred = ridge9.predict(X_test)

print('MSE train: %f, test: %f' % (MSE(y3_train, ridge3_y_train_pred),MSE(y3_test, ridge3_y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y3_train, ridge3_y_train_pred),r2_score(y3_test, ridge3_y_test_pred)))

print('MSE train: %f, test: %f' % (MSE(y6_train, ridge6_y_train_pred),MSE(y6_test, ridge6_y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y6_train, ridge6_y_train_pred),r2_score(y6_test, ridge6_y_test_pred)))

print('MSE train: %f, test: %f' % (MSE(y9_train, ridge9_y_train_pred),MSE(y9_test, ridge9_y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y9_train, ridge9_y_train_pred),r2_score(y9_test, ridge9_y_test_pred)))

plot_residual(y3_train, ridge3_y_train_pred, y3_test, ridge3_y_test_pred)
plot_residual(y6_train, ridge6_y_train_pred, y6_test, ridge6_y_test_pred)
plot_residual(y9_train, ridge9_y_train_pred, y9_test, ridge9_y_test_pred)


from sklearn.neighbors import KNeighborsRegressor
knn3 = KNeighborsRegressor()
knn6 = KNeighborsRegressor()
knn9 = KNeighborsRegressor()
knn3_scores = []
knn6_scores = []
knn9_scores = []

n_neighbors_space= np.arange(1,11)
for n in n_neighbors_space:
    knn3.n_neighbors = n
    knn3_cv_scores = cross_val_score(knn3, X, y3, cv=10)
    knn3_scores.append(np.mean(knn3_cv_scores))
    
    knn6.n_neighbors = n
    knn6_cv_scores = cross_val_score(knn6, X, y6, cv=10)
    knn6_scores.append(np.mean(knn6_cv_scores))
    
    knn9.n_neighbors = n
    knn9_cv_scores = cross_val_score(knn9, X, y9, cv=10)
    knn9_scores.append(np.mean(knn9_cv_scores))

knn3 = KNeighborsRegressor(n_neighbors=n_neighbors_space[np.argmax(knn3_scores)])
print("The best value for n_neighbors is: ", n_neighbors_space[np.argmax(knn3_scores)])
knn3.fit(X_train_std, y3_train)



knn6 = KNeighborsRegressor(n_neighbors=n_neighbors_space[np.argmax(knn6_scores)])
print("The best value for n_neighbors is: ", n_neighbors_space[np.argmax(knn6_scores)])
knn6.fit(X_train_std, y6_train)

knn9 = KNeighborsRegressor(n_neighbors=n_neighbors_space[np.argmax(knn9_scores)])
print("The best value for n_neighbors is: ", n_neighbors_space[np.argmax(knn9_scores)])
knn9.fit(X_train_std, y9_train)

y3_train_pred = knn3.predict(X_train_std)
y3_test_pred = knn3.predict(X_test_std)

y6_train_pred = knn6.predict(X_train_std)
y6_test_pred = knn6.predict(X_test_std)

y9_train_pred = knn9.predict(X_train_std)
y9_test_pred = knn9.predict(X_test_std)

print('3 month: MSE train: %f, test: %f' % (MSE(y3_train, y3_train_pred),MSE(y3_test, y3_test_pred)))
print('3 month: R^2 train: %.3f, test: %.3f' % (r2_score(y3_train, y3_train_pred),r2_score(y3_test, y3_test_pred)))

print('6 month: MSE train: %f, test: %f' % (MSE(y6_train, y6_train_pred),MSE(y6_test, y6_test_pred)))
print('6 month: R^2 train: %.3f, test: %.3f' % (r2_score(y6_train, y6_train_pred),r2_score(y6_test, y6_test_pred)))

print('9 month: MSE train: %f, test: %f' % (MSE(y9_train, y9_train_pred),MSE(y9_test, y9_test_pred)))
print('9 month: R^2 train: %.3f, test: %.3f' % (r2_score(y9_train, y9_train_pred),r2_score(y9_test, y9_test_pred)))

plot_residual(y3_train, y3_train_pred, y3_test, y3_test_pred)
plot_residual(y6_train, y6_train_pred, y6_test, y6_test_pred)
plot_residual(y9_train, y9_train_pred, y9_test, y9_test_pred)

##Feature importances ensembling 
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(max_features=4,
                                learning_rate=0.01,
                                n_estimators=200,
                                subsample=0.6)

gbr.fit(X_train, y3_train)
y3_train_pred = gbr.predict(X_train)
y3_test_pred = gbr.predict(X_test)
print('3 month: MSE train: %f, test: %f' % (MSE(y3_train, y3_train_pred),MSE(y3_test, y3_test_pred)))
print('3 month: R^2 train: %.3f, test: %.3f' % (r2_score(y3_train, y3_train_pred),r2_score(y3_test, y3_test_pred)))


gbr.fit(X_train, y6_train)
y6_train_pred = gbr.predict(X_train)
y6_test_pred = gbr.predict(X_test)
print('6 month: MSE train: %f, test: %f' % (MSE(y6_train, y6_train_pred),MSE(y6_test, y6_test_pred)))
print('6 month: R^2 train: %.3f, test: %.3f' % (r2_score(y6_train, y6_train_pred),r2_score(y6_test, y6_test_pred)))


gbr.fit(X_train, y9_train)
y9_train_pred = gbr.predict(X_train)
y9_test_pred = gbr.predict(X_test)
print('9 month: MSE train: %f, test: %f' % (MSE(y9_train, y9_train_pred),MSE(y9_test, y9_test_pred)))
print('9 month: R^2 train: %.3f, test: %.3f' % (r2_score(y9_train, y9_train_pred),r2_score(y9_test, y9_test_pred)))


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

params_dt ={
             'max_features': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
             'learning_rate': [0.01, 0.1, 0.5],
             'n_estimators': [50, 100, 200, 500],
             'subsample': [0.6, 1],
            }
# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=gbr,
                       param_grid=params_dt,
                       cv=5,
                       n_jobs=-1)

grid_dt.fit(X_train, y3_train)
print(grid_dt.best_params_)
grid3 = grid_dt.best_estimator_

grid_dt.fit(X_train, y6_train)
print(grid_dt.best_params_)
grid6 = grid_dt.best_estimator_

grid_dt.fit(X_train, y9_train)
print(grid_dt.best_params_)
grid9 = grid_dt.best_estimator_

print(grid3.score(X_train, y3_train))
print(grid3.score(X_test, y3_test))
print(grid6.score(X_train, y6_train))
print(grid6.score(X_test, y6_test))
print(grid9.score(X_train, y9_train))
print(grid9.score(X_test, y9_test))


# In[8]:





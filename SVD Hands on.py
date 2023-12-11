import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel(r'D:/Hands on/11_Demension Reduction - SVD/University_Clustering.xlsx')

df.info()

df.describe()

df.columns

df.head()

df.drop(['UnivID'], axis = 1, inplace = True)

df_num = df.select_dtypes(exclude = 'object')

df_cat = df.select_dtypes(include = 'object')

df_num.isnull().sum()

import matplotlib.pyplot as plt
for i in df_num.columns[:]:
    plt.boxplot(df_num[i])
    plt.title('Box plot for ' + str(i))
    plt.show()

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

svd = TruncatedSVD(n_components = 5)
pca = PCA(n_components = 6)

svd_pipe = make_pipeline(SimpleImputer(strategy = 'mean'), StandardScaler(), svd)
pca_pipe = make_pipeline(SimpleImputer(strategy = 'mean'), StandardScaler(), pca)


svd_fit = svd_pipe.fit(df_num)
pca_fit = pca_pipe.fit(df_num)

pca_fit['pca'].components_
svd_fit['truncatedsvd'].components_

pca_transform = pca_fit.transform(df_num)
svd_transform = svd_fit.transform(df_num)

pca_components = pd.DataFrame(pca_fit['pca'].components_, columns = df_num.columns).T
pca_components.columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6']

svd_components = pd.DataFrame(svd_fit['truncatedsvd'].components_, columns = df_num.columns).T
svd_components.columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5']


import joblib

svd_dump = joblib.dump(svd_fit, 'SVD Model')

pca_dump = joblib.dump(pca_fit, 'PCA Model')

import os
os.getcwd()

pca_model = joblib.load('PCA Model')
svd_model = joblib.load('SVD Model')

pca_model_trans = pd.DataFrame(pca_model.transform(df_num))
svd_model_trans = pd.DataFrame(svd_model.transform(df_num))

pca_var = pca_model['pca'].explained_variance_ratio_
svd_var = svd_model['truncatedsvd'].explained_variance_ratio_
import numpy as np
pca_cumvar = np.cumsum(pca_var)
svd_cumvar = np.cumsum(svd_var)

len(pca_cumvar)

from kneed import KneeLocator
k1 = KneeLocator(range(len(svd_cumvar)), svd_cumvar, curve = 'concave', direction = 'increasing')
#k1.elbow
plt.style.use('seaborn')
k1.plot_knee()

k1 = KneeLocator(range(len(svd_cumvar)), svd_cumvar, curve = 'concave', direction = 'increasing')
plt.style.use("seaborn")
plt.plot(range(len(svd_cumvar)), svd_cumvar)
plt.xticks(range(len(svd_cumvar)))
plt.ylabel("Interia")
plt.axvline(x = k1.elbow, color = 'r', label = 'axvline - full height', ls = '--')
plt.show()


final_model = pd.concat([df_cat.Univ, pca_model_trans.iloc[:, 0:3]], axis = 1)
final_model.columns = ['Univ', 'svd1', 'svd2', 'svd3']

ax = final_model.plot('svd1', 'svd2', kind = 'scatter')
final_model[['svd1', 'svd2', 'Univ']].apply(lambda x: ax.text(*x), axis = 1)
final_model[['svd1', 'svd2', 'Univ']].apply(lambda x: ax.test(*x), axis = 1)
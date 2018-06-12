
# coding: utf-8

# ## Nome: Daniel Guimarães Faria
# ## Matricula: 0050010533

# ## 1-Mostre os primeiros registros da tabela

# In[13]:

import pandas as pd
from sklearn.datasets import load_boston
dataset = load_boston()
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data.head()


# ## 2-Observações(linhas) possui nessa base? Quantas Colunas?

# In[15]:

data.shape


# ## 3-Quantas Features Possui?

# Possui 13 ao todo

# ## 4-Qual é o campo Target(respostas) dessa base?

# In[16]:

data['target'] = dataset.target
print(data['target'])


# ## 5-Usando a biblioteca de visualização seaborn, plote o gráfico que mostra a relação entre as features e responses

# In[19]:

import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.pairplot(data, x_vars=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'], y_vars='target', size=7, aspect=0.7, kind='reg')


# ## 6-Prepare X e y usando o pandas

# In[20]:

feature_cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X = data[feature_cols]
X = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
X.head()


# In[25]:

y = data['target']
y = data.target
y.head()


# ## 7-Qual o tipo de dados de X e y?

# In[23]:

print(type(X))
print(X.shape)


# In[26]:

print(type(y))
print(y.shape)


# ## 8-Sobre o que se trata essa base de dados? Que tipo de informações ela guarda?

# É um conjunto de informações coletadas para gerar um banco de dados de avaliação imobiliária de Boston.

# ## 9-Gere um X de treino e y de treino, X de teste e y da base (Split Train/Test)

# In[27]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ## 10-Qual o percentual gerado para criar o conjunto de treino e o conjunto de teste?

# treino - 379*100/506 = 74,901
# teste - 127/506 = 0.2509

# ## 11-Usando modelo de regressão linear do sklearn, treine o modelo com o X e y de treino

# In[28]:

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, y_train)


# ## 12-Imprima os atributos de "intercept" e "coefficients" que foram gerados

# In[29]:

print(linreg.intercept_)
print(linreg.coef_) 


# ## 13-imprima o par "feature names" com os "coefficients"

# In[30]:

list(zip(feature_cols, linreg.coef_))


# ## 14-Faça uma previsão usando o conjunto de X de teste

# In[31]:

y_pred = linreg.predict(X_test)
print(y_pred)


# ## 15-Mostre a margem de error, usando o método "Root Mean Squared Error (RMSE)"

# In[32]:

from sklearn import metrics
import numpy as np
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ## 16-Existe uma forma de melhorar essa margem para que fique menor? Se sim, como seria?

# In[33]:

feature_cols = ['ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X = data[feature_cols]
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




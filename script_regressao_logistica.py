import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

dados = pd.read_excel("Tabela_Unica.xlsx")


X = dados[dados.columns[73:86]]
y = dados[['Numero_Infectados']]


X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.4,random_state=100)


modelo = LogisticRegression()
modelo.fit(X_train, Y_train); #aqui o modelo é construido\


predicao = modelo.predict(X_test)

acc = accuracy_score(Y_test, predicao)
acc = str(round(acc*100,2)) + '%' 

hue = Y_test['Numero_Infectados'] = Y_test['Numero_Infectados'].apply(int)

df = pd.DataFrame({'real': hue, 'predito': predicao})

print(df)
print(acc)

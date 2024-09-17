import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#CARGAR LOS DATOS
train_data=pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#VER CUÁNTOS NANS HAY EN CADA COLUMNA
#print(train_data.isnull().sum())

#SUBSTITUIR LOS NANS DE LA COLUMNA "Fare" POR LA MEDIA DE LA MISMA COLUMNA (SOLO AFECTA A LOS DATOS DE VALIDACIÓN (test_data))
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

#SUBSTITUIR LOS NANS DE LA COLUMNA "Age" POR LA MEDIA DE LA MISMA COLUMNA
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)

#SUBSTITUIR LOS NANS DE LA COLUMNA "Embarked" POR LA MEDIANA DE LA MISMA COLUMNA
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0],inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0],inplace=True)

#TRATAMOS LAS VARIABLES CATEGÓRICAS
#CREAMOS EL DICCIONARIO DE CORRESPONDENCIAS
labelsDict={'Sex':{'female':0,'male':1},'Embarked':{'S':0,'C':1,'Q':2}}
#APLICAMOS LA TÉCNICA LABEL ENCODING (SUBSTITUIR STRINGS POR NÚMEROS)
train_data['Sex']=train_data['Sex'].map(labelsDict['Sex'])
train_data['Embarked']=train_data['Embarked'].map(labelsDict['Embarked'])
test_data['Sex']=test_data['Sex'].map(labelsDict['Sex'])
test_data['Embarked']=test_data['Embarked'].map(labelsDict['Embarked'])

#SELECCIONAMOS LAS CARACTERÍSTICAS QUE NOS INTERESAN ('PassengerId','Name','Ticket' Y 'Cabin' NO NOS INTERESAN PORQUE NO AFECTAN AL TARGET)
features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

#DIVIDIMOS LOS DATOS (CARACTERÍSTICAS Y OBJETIVOS)
X=train_data[features]
Y=train_data['Survived']
X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.15,random_state=42)

#INSTANCIAMOS EL MODELO Y LO ENTRENAMOS
model=LogisticRegression(max_iter=500)
model.fit(X_train,Y_train)

#EVALUAMOS EL MODELO (PRECISIÓN: 80%)
Y_prediction=model.predict(X_val)
accuracy=accuracy_score(Y_val,Y_prediction)
#print(accuracy)

#VOLVEMOS A ENTRENAR EL MODELO PERO AHORA CON TODOS LOS DATOS
model.fit(X,Y)

#PREDECIMOS PARA LOS DATOS DE VALIDACIÓN ("test_data") SI LOS PASAJEROS SOBREVIVIRÍAN O NO
predictions=model.predict(test_data[features])

#ACTUALIZAMOS EL ARCHIVO DE LOS DATOS DE VALIDACIÓN (AÑADIMOS LA COLUMNA "Survived" QUE CONTIENE NUESTRAS PREDICCIONES)
test_data['Survived']=predictions

#GENERAMOS EL ARCHIVO FINAL (SOLO ID DE PASAJERO Y SI SOBREVIVIRÍA O NO SEGÚN NUESTRA PREDICCIÓN)
submission=test_data[['PassengerId','Survived']]
submission.to_csv('my_submission.csv',index=False)


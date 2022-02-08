

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/Crop_recommendation.csv')

#Passando nome para portugues
df['label'] = df['label'].map({
                              'rice':'Arroz', 
                              'maize':'Milho',
                              'chickpea':'Grão de Bico',
                              'kidneybeans':'Feijão roxo',
                              'pigeonpeas':'Feijão bóer',
                              'mothbeans': 'Feijão',
                              'mungbean':'Feijão Mungo',
                              'blackgram':'Feijão da India',
                              'lentil':'Lentilha',
                              'pomegranate':'Romã',
                              'banana':'Banana',
                              'mango':'Manga',
                              'grapes':'Uva',
                              'watermelon':'Melancia',
                              'muskmelon':'Melão',
                              'apple':'Maçã',
                              'orange':'Laranja',
                              'papaya':'Mamão Papaia',
                              'coconut':'Coco',
                              'cotton':'Algodão',
                              'jute':'Fibra Vegetal',
                              'coffee':'Café'
                              },
                                na_action=None)

df.isnull().sum()

df.info()

df.describe(include='all')

#Correlação entre as colunas
plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot=True,cmap='inferno_r')
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Dividindo os dados em teste e treino
X_train, X_test, y_train, y_test = train_test_split(df.drop('label',axis=1),df['label'],test_size=0.2)

#Instanciando o Classificador
clf = DecisionTreeClassifier()

#Treinando o modelo
clf.fit(X_train, y_train)

#Fazendo a predição
y_pred = clf.predict(X_test)

#Acuracia
acc = accuracy_score(y_test,y_pred)
acc

#Parametros do solo
def range_parameter(t):
  df1 = df[df['label'] == str(t)]
  n_max = round(df1['N'].max(), 2)
  n_min = round(df1['N'].min(), 2)
  p_max = round(df1['P'].max(), 2)
  p_min = round(df1['P'].min(), 2)
  temp_max = round(df1['temperature'].max(), 2)
  temp_min = round(df1['temperature'].min(), 2)
  hum_max = round(df1['humidity'].max(), 2)
  hum_min = round(df1['humidity'].min(), 2)
  k_max = round(df1['K'].max(), 2)
  k_min = round(df1['K'].min(), 2)
  ph_max = round(df1['ph'].max(), 2)
  ph_min = round(df1['ph'].min(), 2)
  rain_max = round(df1['rainfall'].max(), 2)
  rain_min = round(df1['rainfall'].min(), 2)
  return(f'Intervalo das caracteristicas do solo para {t}: \
  \nNitrogênio: {n_min}% á {n_max}%\
  \nFósforo: {p_min}% á {p_max}%\
  \nPotássio: {k_min}% á {k_max}%\
  \nTemperatura: {temp_min}°C á {temp_max}°C\
  \nHumidade: {hum_min}% á {hum_max}%\
  \nPH: {ph_min} á {ph_max}\
  \nChuva:{rain_min}mm á {rain_max}mm\n' )

#Entrada do usuario
N = int(input('Nitrogênio: '))
P = int(input('Fósforo: '))
K = int(input('Potássio: '))
T = float(input('Temperatura: '))
H = float(input('Humidade: '))
PH = float(input('PH: '))
R = float(input('Chuva: '))

#Passando as entradas para um Dicionario
dic = {'N': [N],'P': [P], 'K': [K], 'temperature': [T],'humidity': [H], 'ph': [PH], 'rainfall': [R]}

#Transformando o dicionario em um Dataframe
df_predic = pd.DataFrame(dic)

#Passando o DataFrame para predição
t = clf.predict(df_predic)

#Saida prevista pelo sistema
t = str(t).replace("'", '').replace("'", '').replace('[', '').replace(']', '')

print(f'Melhor semente para o terreno é {t}\n')

r = range_parameter(t)
print(r)


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy

import sklearn
from sklearn.preprocessing import StandardScaler


# Carregar o dataset
hcc_data = pd.read_csv(r'C:\Users\leono\OneDrive\Documents\Faculdade\EIACD\PROJETO2\hcc_dataset.csv')


# Substituir '?' por NaN
hcc_data.replace('?', np.nan, inplace=True)

def impute_missing_data(data):
    for column in data.columns:
        if data[column].dtype == 'object':  # Para dados categóricos
            # Preencher com a moda
            mode_value = data[column].mode()[0]
            data[column].fillna(mode_value,inplace=True)
        else:  # Para dados numéricos
            # Preencher com a mediana
            median_value = data[column].median()
            data[column].fillna(median_value,inplace=True)

# Aplicar a função ao dataset
impute_missing_data(hcc_data)

for col in hcc_data.columns:
    if hcc_data[col].dtype == object :
        # Padronizar para maiúsculas e remover espaços
        hcc_data[col] = hcc_data[col].str.upper().str.strip()
        # Substituições
        hcc_data[col].replace({'YES': 1, 'NO': 0}, inplace=True)
        hcc_data[col].replace({'FEMALE':1 ,'MALE':0},inplace=True)
        hcc_data[col].replace({'LIVES':1 ,'DIES':0},inplace=True)

# Lista de colunas que devem ser numéricas
numerical_cols = ['Grams_day', 'Packs_year', 'INR', 'AFP', 'Hemoglobin', 'MCV', 'Leucocytes', 'Platelets',
                  'Albumin', 'Total_Bil', 'ALT', 'AST', 'GGT', 'ALP', 'TP', 'Creatinine', 'Nodules', 'Major_Dim',
                  'Dir_Bil', 'Iron', 'Sat', 'Ferritin','Class']

# Converter cada coluna para numérico, tratando erros
for col in numerical_cols:
    # Verificar se a coluna é de tipo object, indicando possível presença de strings
    if hcc_data[col].dtype == 'object':
        hcc_data[col]=pd.to_numeric(hcc_data[col].str.replace(' ', ''), errors='coerce')
    else:
        # Se não for object, converte diretamente
        hcc_data[col]=pd.to_numeric(hcc_data[col], errors='coerce')

print(hcc_data.head())
from sklearn.model_selection  import train_test_split

X = hcc_data.drop(['Class'],axis = 1) #sem a coluna Class
y = hcc_data['Class'] #apenas coluna Class

X_train,X_test,y_train,Y_test = train_test_split(X,y,test_size=0.2)

# Criando um novo DataFrame apenas com colunas numéricas
numeric_data = hcc_data.select_dtypes(include=[np.number])

# Calculando a correlação apenas nas colunas numéricas
correlation_matrix = numeric_data.corr()

# Criando o heatmap
plt.figure(figsize=(12,10 ))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Coefficient of Correlation'},annot_kws={'size':8})
plt.title('Heatmap de Correlação das Variáveis Numéricas')
#plt.show()

# Criar histogramas para todas as variáveis numéricas
for column in hcc_data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(hcc_data[column], kde=True, color='blue')  # kde=True adiciona a curva de densidade
    plt.title(f'Distribuição de {column}')
    plt.xlabel(column)
    plt.ylabel('Frequência')
    #plt.show()

# Loop para criar um gráfico de barras para cada variável categórica
for column in hcc_data.columns:
    if hcc_data[column].dtype == 'object':  # Supondo que todas as categóricas sejam 'object'
        plt.figure(figsize=(5, 3))  # Configura o tamanho da figura
        sns.countplot(x=column, data=hcc_data, palette='pastel')  # Cria o gráfico de barras
        plt.title(f'Distribuição de {column}')  # Define o título do gráfico
        plt.xlabel(column)  # Define o label do eixo x
        plt.ylabel('Frequência')  # Define o label do eixo y
        plt.xticks(rotation=45)  # Rotaciona os labels do eixo x para melhor visualização, se necessário
        #plt.show()  # Exibe o gráfico

#Data preprocessing
train_data = X_train.join(y_train) #junta apenas para  visualizar os dados de treino
train_data['Age'] = np.log(train_data['Age']+1) 













  

# Exibir as primeiras linhas do dataset
#print(hcc_data.head())

#print(hcc_data.Sat.value_counts())

# Resumir as informações do dataset
#print(hcc_data.info())

# Descrição estatística básica
#print(hcc_data.describe())


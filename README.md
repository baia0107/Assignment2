# Assigment 2: Projeto de Machine Learning para Predição de Sobrevivência de Pacientes com HCC

## Descrição do projeto

Este projeto aplica técnicas de Machine Learning para prever a sobrevivência de pacientes com Carcinoma Hepatocelular (HCC). Utilizamos um conjunto de dados que contém diversas características dos pacientes e implementamos vários modelos para a previsão.

## Estrutura do Projeto

- `final-Copy5.ipynb/`: Contém o código-fonte principal do projeto.
- `README.md`: Este arquivo contém a folha de instruções
- `hcc_data.csv`: Contém o arquivo de dados

## Requisitos

- Python 3.x: Versão do Python necessária
- Pacotes/bibliotecas: pandas, scikit-learn, numpy,seaborn, scipy

## Instruções de Instalação

1. Clone o repositório: `git clone final-Copy5.ipynb`
2. Instale as dependências: `pip install -r # aqui incluir os pacotes

## Instruções de Uso

Execute o script `final-Copy5.ipynb` e siga as instruções na tela.

## Uso

### 1. Pré-processamento dos Dados

O conjunto de dados é carregado e algumas novas características são criadas:

```python
import pandas as pd

# Carregar os dados
hcc_data = pd.read_csv('data/hcc_data.csv')


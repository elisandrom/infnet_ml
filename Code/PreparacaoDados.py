import mlflow
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

##---------------------------------------------------------------------------
warnings.filterwarnings('ignore')
randonState = 13
percentualTeste = 20 #Exemplo: 20 equivale a 20%
#Montando o experimento para o MLFlow
#mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Kobe Bryant - Elisandro")
##---------------------------------------------------------------------------

#Lendo o parquet de DEV
df = pd.read_parquet("Data/Raw/dataset_kobe_dev.parquet")

#Colunas importantes da base
columnasImportantes = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']

#Filtra os registros que tem a definição do shot_made_flag
df_filtrado = df[df['shot_made_flag'].notnull()]

#Filtra somente os arremeços de 2 pontos
df_filtrado = df_filtrado[df_filtrado['shot_type'] == '2PT Field Goal']

#Deixa no dataframe somente as colunas importantes
df_dados = df_filtrado[columnasImportantes]

#Salva o dataframe filtrado
df_dados.to_parquet("Data/Processed/data_filtered.parquet")

##---------------------------------------------------------------------------
#print(df["shot_type"].unique())
#print(df_dados.columns)
print(f"Dimensão do DataFrame Original: {len(df)}")
print(f"Dimensão do DataFrame Filtrado: {len(df_dados)}")
##---------------------------------------------------------------------------

#Início da separação dos dados
X = df_dados.copy()
Y = df_filtrado[['shot_made_flag']]

#print(X.shape)
#print(Y.shape)

#Separando treino e teste - com estratificação 80%-20%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=(percentualTeste/100), stratify=Y, random_state=randonState)

#Salva os dataframes de treino e teste
x_train.join(y_train).to_parquet("Data/Processed/base_train.parquet")
x_test.join(y_test).to_parquet("Data/Processed/base_test.parquet")

##---------------------------------------------------------------------------
#Criando o PreparacaoDados
##***************************************************************************
#Iniciando uma run do MlFlow para o pipeline de preparação de dados
with mlflow.start_run(run_name='PreparacaoDados'):
    mlflow.log_param("teste_percentual", percentualTeste)
    mlflow.log_metric("base_treino_tamanho", len(x_train))
    mlflow.log_metric("base_teste_tamanho", len(x_test))


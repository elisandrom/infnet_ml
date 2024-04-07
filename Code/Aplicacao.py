import mlflow
import warnings
import pandas as pd
import requests
from sklearn.metrics import log_loss, f1_score

##---------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#Montando o experimento para o MLFlow
#mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Kobe Bryant - Elisandro")
##---------------------------------------------------------------------------

#Antes de tudo, precisa executar no console:
#mlflow models serve -m "runs:/f477f04534fd44cb9612ba77f2b31b7d/model_decision_tree" --no-conda -p 8081

##---------------------------------------------------------------------------

# Carregando a base de produção
df = pd.read_parquet("Data/Raw/dataset_kobe_prod.parquet")

dfOriginal = df.copy()
dfOriginal = dfOriginal[dfOriginal['shot_made_flag'].notnull()]

#Colunas importantes da base
columnasImportantes = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']

#Filtra os registros que tem a definição do shot_made_flag
df = df[df['shot_made_flag'].notnull()]

#Deixa no dataframe somente as colunas importantes
df = df[columnasImportantes]

#print(df)

#Transforma os dados do dataframe em json
dados_json = df.to_json(orient='split')

#Faz a requisição localmente para a API do MLFlow
responseMLFlow = requests.post(
    'http://127.0.0.1:8081/invocations',
    headers={'Content-Type':'application/json'},
    json={
        "dataframe_split": 
        {
            "columns": df.columns.tolist(),
            "data": df.values.tolist()
        }
    }
)

# Verificar a resposta for válida = 200
if responseMLFlow.status_code == 200:
    print("[OK] Dados recebidos do MLFlow")
    predicoes = responseMLFlow.json()
    
    #print(predicoes)

    # Calcula as métricas
    log_loss = log_loss(dfOriginal['shot_made_flag'].values, predicoes['predictions'])
    f1_score = f1_score(dfOriginal['shot_made_flag'].values, predicoes['predictions'])
    
    print(f"Predição Log Loss: {log_loss}")
    print(f"Predição F1 Score: {f1_score}")

    #Monta o dataframe com os resultados
    #df_results = pd.DataFrame({"target": dfOriginal["shot_made_flag"], "prediction": predicoes['predictions']})
    df_results = dfOriginal.copy()
    df_results['Prediction'] = predicoes['predictions']
    #df_results.insert(len(df_results.columns), "Prediction", predicoes['predictions'])
    #print(df_results)

    #Agora armazenar no MLFlow
    with mlflow.start_run(run_name="PipelineAplicacao"):
        # Registro das métricas
        mlflow.log_metric("log_loss", log_loss)
        mlflow.log_metric("f1_score", f1_score)

        mlflow.set_tag('model', 'Decision Tree Classifier')

        #print(mlflow.active_run().info.artifact_uri)

        #Salvando o dataframe com os resultados no mlflow
        df_results.to_parquet(mlflow.active_run().info.artifact_uri + "/result.parquet")
    
else:
    print("[ERRO] Motivo:", responseMLFlow.text)
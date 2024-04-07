import streamlit as st
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

nomeExperimento = "Kobe Bryant - Elisandro"

#Montando o experimento para o MLFlow
#mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment(nomeExperimento)

#Título do dashboard
st.title('Monitoramento do Modelo')

#Seção de experimento
st.header('Experimento: ' + nomeExperimento)

#------------------------------------------------------------------------------------------------
#Busca as informações do experimento
experiment = mlflow.get_experiment_by_name(nomeExperimento)
experiment_id = experiment.experiment_id
runs = mlflow.search_runs(experiment_ids=[experiment_id])
runs['experiment_name'] = nomeExperimento
runs = runs[runs['experiment_name'] == nomeExperimento]

#print(runs.columns)

st.write('ID no MLFlow:', experiment_id)
st.write(''.ljust(10, '-'))

st.header('Histórico de Runs')
st.write('Resumo:', runs)
st.write(''.ljust(10, '-'))

#------------------------------------------------------------------------------------------------
#Busca o run final que possui o parquet com o resultado
st.header('PipelineAplicacao')
runPipelineAplicacao = runs[runs['tags.mlflow.runName'] == 'PipelineAplicacao']
#st.write(runPipelineAplicacao)

runIdPipelineAplicacao = runPipelineAplicacao['run_id'].values[0]
st.markdown('<b>ID:</b> ' + runIdPipelineAplicacao, unsafe_allow_html=True)
st.markdown(f"<b>Modelo:</b> {runPipelineAplicacao['tags.model'].values[0]}", unsafe_allow_html=True)

st.markdown(f"<b>Métrica - F1 Score:</b> {runPipelineAplicacao['metrics.f1_score'].values[0]}", unsafe_allow_html=True)
st.markdown(f"<b>Métrica - Log Loss:</b> {runPipelineAplicacao['metrics.log_loss'].values[0]}", unsafe_allow_html=True)

#Pega o local do arquivo final salvo
#artifact_uri = mlflow.get_run(runIdPipelineAplicacao).info.artifact_uri
uriFile = runPipelineAplicacao['artifact_uri'].values[0] + "/result.parquet"

#Carrega o parquet do resultado
dfResultado = pd.read_parquet(uriFile)

#Escreve na página o arquivo final
st.markdown('<b><u>Resultado Final</b></u>', unsafe_allow_html=True)
st.write(dfResultado)
st.write(f"Total de Registros: {len(dfResultado)}")

#------------------------------------------------------------------------------------------------

st.header('Demonstração dos Arremessos')
st.write('Base Original x Predição')

fig, axes = plt.subplots(2, 2, figsize=(10,14))

axes[0][0].set_title('Base Original', fontsize=10)
axes[0][0].set_xlabel("Errou o Arremesso", fontsize=10)
axes[0][0].scatter(dfResultado.loc[dfResultado['shot_made_flag'] == 0].loc_x, dfResultado.loc[dfResultado['shot_made_flag'] == 0].loc_y, color='red', alpha=0.05)
axes[0][0].set_xlim(dfResultado.loc[dfResultado['shot_made_flag'] == 0].loc_x.min(), dfResultado.loc[dfResultado['shot_made_flag'] == 0].loc_x.max())
axes[0][0].set_ylim(dfResultado.loc[dfResultado['shot_made_flag'] == 0].loc_y.min(), dfResultado.loc[dfResultado['shot_made_flag'] == 0].loc_y.max())

axes[0][1].set_title('Predição', fontsize=10)
axes[0][1].set_xlabel("Errou o Arremesso", fontsize=10)
axes[0][1].scatter(dfResultado.loc[dfResultado['Prediction'] == 0].loc_x, dfResultado.loc[dfResultado['Prediction'] == 0].loc_y, color='orange', alpha=0.05)
axes[0][1].set_xlim(dfResultado.loc[dfResultado['shot_made_flag'] == 0].loc_x.min(), dfResultado.loc[dfResultado['shot_made_flag'] == 0].loc_x.max())
axes[0][1].set_ylim(dfResultado.loc[dfResultado['shot_made_flag'] == 0].loc_y.min(), dfResultado.loc[dfResultado['shot_made_flag'] == 0].loc_y.max())

axes[1][0].set_title('Base Original', fontsize=10)
axes[1][0].set_xlabel("Acertou o Arremesso", fontsize=10)
axes[1][0].scatter(dfResultado.loc[dfResultado['shot_made_flag'] == 1].loc_x, dfResultado.loc[dfResultado['shot_made_flag'] == 1].loc_y, color='blue', alpha=0.05)
axes[1][0].set_xlim(dfResultado.loc[dfResultado['shot_made_flag'] == 1].loc_x.min(), dfResultado.loc[dfResultado['shot_made_flag'] == 1].loc_x.max())
axes[1][0].set_ylim(dfResultado.loc[dfResultado['shot_made_flag'] == 1].loc_y.min(), dfResultado.loc[dfResultado['shot_made_flag'] == 1].loc_y.max())

axes[1][1].set_title('Predição', fontsize=10)
axes[1][1].set_xlabel("Acertou o Arremesso", fontsize=10)
axes[1][1].scatter(dfResultado.loc[dfResultado['Prediction'] == 1].loc_x, dfResultado.loc[dfResultado['Prediction'] == 1].loc_y, color='green', alpha=0.05)
axes[1][1].set_xlim(dfResultado.loc[dfResultado['shot_made_flag'] == 1].loc_x.min(), dfResultado.loc[dfResultado['shot_made_flag'] == 1].loc_x.max())
axes[1][1].set_ylim(dfResultado.loc[dfResultado['shot_made_flag'] == 1].loc_y.min(), dfResultado.loc[dfResultado['shot_made_flag'] == 1].loc_y.max())

st.pyplot(fig)

#------------------------------------------------------------------------------------------------
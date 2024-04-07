## Ordem de execução por dentro do Visual Code:
A. Terminal -> New Terminal e executar: mlflow server --host 127.0.0.1 --port 8080

B. Executar o script: PreparacaoDados.py

C. Executar o script: Treinamento.py

D. Dentro do UI do MLFlow, pegar o ID do Run "Treinamento" do model_decision_tree

E. Executar o comando para servir o modelo por API da MLFlow
    
    Visual Code -> Terminal -> New Terminal e executar: 

    mlflow models serve -m "runs:/{run_id da etapa anterior}/model_decision_tree" --no-conda -p 8081

F. Executar o script: Aplicacao.py

G. Executar o comando para iniciar o Streamlit:
   
    Visual Code -> Terminal -> New Terminal e executar:
   
    streamlit run c:/Users/Elisandro/Documents/trabalho_disciplina4/Code/Dashboard.py

H. Automaticamente vai abrir a página do Streamlit

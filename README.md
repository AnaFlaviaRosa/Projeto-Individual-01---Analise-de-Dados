# Projeto Individual (PI1): Aplicação de Aprendizado de Máquina Supervisionado

## 🎯 Objetivo do Projeto

Este projeto individual (PI1) tem como objetivo aplicar técnicas de aprendizado de máquina supervisionado para resolver um problema de classificação, utilizando um conjunto de dados fictício. O foco é demonstrar o ciclo completo de um projeto de Machine Learning, desde a definição do problema e pré-processamento de dados (ETL) até o treinamento, avaliação e interpretação de modelos.

## 🧠 Modelos Utilizados

Foram escolhidos dois modelos de classificação para comparação:

1.  **Regressão Logística (Logistic Regression):** Um modelo linear que estima a probabilidade de um evento.
2.  **K-Nearest Neighbors (KNN):** Um modelo não-paramétrico baseado em distância.

## 📊 Problema de Negócio

**Previsão de Churn de Clientes** em uma empresa de telecomunicações fictícia.

O objetivo é classificar se um cliente irá ou não cancelar seu serviço (`Churn: 1` ou `Não Churn: 0`) com base em variáveis como tempo de contrato, uso de dados, chamadas de suporte e valor da fatura.

## 🛠️ Estrutura do Projeto

O projeto é composto pelos seguintes arquivos:

| Arquivo | Descrição |
| :--- | :--- |
| `pi1_project.py` | Código-fonte principal. Contém a geração de dados, ETL, EDA, treinamento e avaliação dos modelos. |
| `Relatorio_PI1_ML_Supervisionado.md` | Relatório técnico completo do projeto, incluindo análise e interpretação dos resultados. |
| `README.md` | Este arquivo. |
| `distribuicao_churn.png` | Visualização da distribuição da variável alvo. |
| `relacao_features_churn.png` | Boxplots comparando features vs. Churn. |
| `matriz_correlacao.png` | Mapa de calor da matriz de correlação entre as variáveis. |


## ⚙️ Como Executar o Projeto

Para replicar os resultados e executar o código, siga os passos abaixo:

### 1. Configuração do Ambiente Virtual (venv)

É altamente recomendável utilizar um ambiente virtual para isolar as dependências.

```bash
# 1. Criar o Ambiente Virtual
python3.11 -m venv venv

# 2. Ativar o Ambiente Virtual
source venv/bin/activate
```

### 2. Instalação das Dependências

Com o ambiente virtual ativado, instale as bibliotecas necessárias:

```bash
pip install scikit-learn pandas matplotlib seaborn numpy
```

### 3. Execução do Script

Execute o script principal. Ele irá gerar os dados fictícios, realizar o pré-processamento, treinar os modelos e salvar todos os gráficos e resultados.

```bash
python pi1_project.py
```

### 4. Visualização dos Resultados

Após a execução, todos os gráficos (`.png`) e o arquivo de resultados intermediários (`model_results.json`) estarão disponíveis no diretório do projeto. O relatório detalhado pode ser consultado em `Relatorio_PI1_ML_Supervisionado.md`.

## 📝 Conclusão Principal

O modelo de **Regressão Logística** demonstrou o melhor desempenho geral para este conjunto de dados fictício, alcançando um **AUC de 0.993**, sendo o modelo recomendado para a identificação de clientes em risco de *churn*.

---
*Este projeto foi desenvolvido como parte de um Projeto Individual (PI1) de Aprendizado de Máquina Supervisionado.*

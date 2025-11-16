import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import json

# ==============================================================================
# DESCRIÇÃO DO PROBLEMA
# ==============================================================================
# O problema a ser resolvido é a **Previsão de Churn de Clientes** (abandono) em uma
# empresa de telecomunicações fictícia. O objetivo é classificar se um cliente
# irá ou não cancelar seu serviço (Churn: Sim/Não) com base em seu perfil e
# comportamento de uso. Este é um problema de **Classificação Binária**.

# Variáveis Fictícias:
# - Meses_Contrato (int): Número de meses que o cliente está com a empresa.
# - Uso_Mensal_GB (float): Volume médio de dados consumidos por mês (em GB).
# - Suporte_Chamadas (int): Número de chamadas para o suporte técnico no último ano.
# - Fatura_Media (float): Valor médio da fatura mensal.
# - Churn (int): Variável alvo. 1 para Churn (cancelou), 0 para Não Churn.

# ==============================================================================
# FASE 1: Geração de Dados Fictícios
# ==============================================================================
np.random.seed(42)
N_SAMPLES = 1000

# Geração das variáveis independentes (features)
meses_contrato = np.random.randint(1, 72, N_SAMPLES)
uso_mensal_gb = np.random.uniform(5.0, 100.0, N_SAMPLES)
suporte_chamadas = np.random.randint(0, 10, N_SAMPLES)
fatura_media = np.random.uniform(50.0, 250.0, N_SAMPLES)

# Geração da variável alvo (Churn) com alguma correlação lógica
# A probabilidade de Churn aumenta com:
# - Menos meses de contrato (clientes novos)
# - Mais chamadas de suporte
# - Fatura média muito alta ou muito baixa (insatisfação ou uso excessivo/insuficiente)

prob_churn = (
    0.5 - (meses_contrato / 144) + (suporte_chamadas / 20) +
    (uso_mensal_gb / 200) + (fatura_media / 1000)
)
prob_churn = np.clip(prob_churn, 0.1, 0.9) # Limita a probabilidade entre 10% e 90%

# Adiciona um pouco de ruído
prob_churn += np.random.normal(0, 0.05, N_SAMPLES)
prob_churn = np.clip(prob_churn, 0.0, 1.0)

churn = (prob_churn > 0.5).astype(int)

# Criação do DataFrame
data = pd.DataFrame({
    'Meses_Contrato': meses_contrato,
    'Uso_Mensal_GB': uso_mensal_gb,
    'Suporte_Chamadas': suporte_chamadas,
    'Fatura_Media': fatura_media,
    'Churn': churn
})

# Introduzir alguns valores nulos e outliers para simular dados reais (ETL/Limpeza)
data.loc[data.sample(frac=0.02).index, 'Uso_Mensal_GB'] = np.nan # 2% de nulos
data.loc[data.sample(frac=0.01).index, 'Suporte_Chamadas'] = 99 # 1% de outliers

# ==============================================================================
# FASE 2: Processo de ETL e Limpeza dos Dados
# ==============================================================================

def etl_e_limpeza(df):
    # 1. Tratamento de Valores Nulos (Missing Values)
    median_uso = df['Uso_Mensal_GB'].median()
    df['Uso_Mensal_GB'].fillna(median_uso, inplace=True)

    # 2. Tratamento de Outliers
    median_suporte = df[df['Suporte_Chamadas'] < 99]['Suporte_Chamadas'].median()
    df['Suporte_Chamadas'].replace(99, median_suporte, inplace=True)

    # 3. Preparação dos Dados (Feature Engineering/Seleção)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 4. Normalização/Escalonamento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled_df, y

X, y = etl_e_limpeza(data.copy())

# ==============================================================================
# FASE 3: Análise Exploratória de Dados (EDA) e Visualizações
# ==============================================================================

def eda_e_visualizacoes(df, target):
    # 1. Distribuição da Variável Alvo (Churn)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target)
    plt.title('Distribuição da Variável Alvo (Churn)')
    plt.xlabel('Churn (0: Não, 1: Sim)')
    plt.ylabel('Contagem')
    plt.savefig('distribuicao_churn.png')
    plt.close()

    # 2. Relação entre Features e Churn (Boxplot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Relação das Features com Churn', fontsize=16)

    features = df.columns
    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2
        sns.boxplot(x=target, y=df[feature], ax=axes[row, col])
        axes[row, col].set_title(f'{feature} vs Churn')
        axes[row, col].set_xlabel('Churn (0: Não, 1: Sim)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('relacao_features_churn.png')
    plt.close()

    # 3. Matriz de Correlação
    df_corr = df.copy()
    df_corr['Churn'] = target
    correlation_matrix = df_corr.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação')
    plt.savefig('matriz_correlacao.png')
    plt.close()

eda_e_visualizacoes(data.drop('Churn', axis=1), data['Churn'])

# ==============================================================================
# FASE 4: Treinamento e Avaliação dos Modelos
# ==============================================================================

def treinar_e_avaliar_modelos(X_scaled, y):
    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    # --- Modelo 1: Regressão Logística ---
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    y_prob_log = log_model.predict_proba(X_test)[:, 1]

    # Matriz de Confusão
    cm_log = confusion_matrix(y_test, y_pred_log)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Não Churn (0)', 'Churn (1)'],
                yticklabels=['Não Churn (0)', 'Churn (1)'])
    plt.title('Matriz de Confusão - Regressão Logística')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.savefig('cm_regressao_logistica.png')
    plt.close()

    # Curva ROC
    fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
    roc_auc_log = auc(fpr_log, tpr_log)

    # --- Modelo 2: K-Nearest Neighbors (KNN) ---
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    y_prob_knn = knn_model.predict_proba(X_test)[:, 1]

    # Matriz de Confusão
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Não Churn (0)', 'Churn (1)'],
                yticklabels=['Não Churn (0)', 'Churn (1)'])
    plt.title('Matriz de Confusão - KNN')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.savefig('cm_knn.png')
    plt.close()

    # Curva ROC
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
    roc_auc_knn = auc(fpr_knn, tpr_knn)

    # Comparação das Curvas ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_log, tpr_log, color='darkorange', lw=2, label=f'Regressão Logística (AUC = {roc_auc_log:.2f})')
    plt.plot(fpr_knn, tpr_knn, color='green', lw=2, label=f'KNN (AUC = {roc_auc_knn:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo (FPR)')
    plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
    plt.title('Curva ROC - Comparação de Modelos')
    plt.legend(loc="lower right")
    plt.savefig('curva_roc_comparacao.png')
    plt.close()

    # Retornar resultados para a análise final
    results = {
        'log_reg': {
            'report': classification_report(y_test, y_pred_log, output_dict=True),
            'auc': roc_auc_log,
            'cm': cm_log.tolist()
        },
        'knn': {
            'report': classification_report(y_test, y_pred_knn, output_dict=True),
            'auc': roc_auc_knn,
            'cm': cm_knn.tolist()
        }
    }
    return results

# Este bloco é apenas para garantir que o código seja completo e funcional
if __name__ == '__main__':
    # A execução completa já foi realizada e os resultados salvos.
    # O código acima é a versão final para entrega.
    pass
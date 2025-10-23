<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=31688e&height=120&section=header"/>

[![Typing SVG](https://readme-typing-svg.herokuapp.com/?color=31688e&size=35&center=true&vCenter=true&width=1000&lines=Balanço+de+Dados+e+Técnicas+Preditivas;A+base+agora+ganha+equilíbrio+e+significado;Normalização,+codificação+e+amostragem+inteligente;Modelagem+de+alto+desempenho!)](https://git.io/typing-svg)

---
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue?logo=plotly)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Graphics-lightblue?logo=seaborn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Graphs-purple?logo=plotly)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Preprocessing-green?logo=scikitlearn)
![Imbalanced Learn](https://img.shields.io/badge/Imbalanced%20Learn-SMOTE-yellow?logo=python)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

<details>

<summary>📚 Sumário</summary>

## Sumário do Projeto

- [🎯 Objetivo](#objetivo)
- [🔗Jornada do Projeto](#jornada-do-projeto)
- [📂 Estrutura do Projeto](#-estrutura-do-projeto)
- [🧾 Dicionário de Dados](#-dicionário-de-dados)
- [🧩 Etapa 1 — Pré Processamento](#etapa-1---pré-processamento)
- [📊 Etapa 2 — Análise Univariada e Bivariada](#-etapa-2---análise-univariada-e-bivariada)
- [📈 Etapa 3 — Correlação Balanceamento e Codificação](#-etapa-3---correlação-balanceamento-e-codificação)
- [🧭 Próximos Passos (Parte 2)](#-próximos-passos-parte-2)
- [💭 Reflexão Final](#-reflexão-final)
- [👨‍💻Autor](#autor)
- [📦 Instalação dos Requisitos](#-instalação-dos-requisitos)

</details>

# 🧮 Projeto de Credit Score - Parte 1 

> Projeto Credit Score - Parte 1
Nesta primeira etapa do projeto Credit Score, construímos uma base sólida para compreender o perfil dos clientes e preparar os dados para modelos preditivos de crédito. O foco é criar um pipeline de dados limpo, balanceado e estatisticamente confiável — essencial para análises robustas e machine learning.
> O objetivo é preparar e compreender a base de clientes antes da modelagem, aplicando técnicas de pré-processamento, análise univariada e bivariada, e balanceamento de classes.

---

### Objetivo

O termo Credit Score refere-se a uma pontuação numérica que indica a credibilidade de um indivíduo em relação ao cumprimento de suas obrigações financeiras — como empréstimos e cartões de crédito.  

O objetivo deste projeto é prever o risco de inadimplência de clientes com base em atributos demográficos e financeiros, preparando os dados para uma futura modelagem preditiva.

---

### 📂 Estrutura do Projeto  
```markdown

📁 credit_score_part1/
│
├── data/                         # Base de dados original e tratada
├── img/                          # Gráficos gerados nas análises
├── notebooks/                    # Notebooks de processamento
│   └── credit_score_parte1.ipynb
├── README.md                     # Este arquivo
└── requirements.txt              # Dependências do projeto


````

### 🔗Jornada do Projeto
<details>
<summary><b>🔗 Jornada do Projeto</b></summary>

```markdown
| Etapa                                | Descrição                                                                                                                                                                                                                            |
| -------------------------------------| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pré-processamento dos Dados          | Incluiu limpeza, normalização, padronização e verificação de missing values, assegurando consistência e qualidade na base final.                                                                                                       |
| Análise Univariada                   | Exploramos individualmente cada variável, identificando distribuições, outliers e possíveis inconsistências. Essa etapa permitiu entender o comportamento isolado dos atributos e detectar oportunidades de normalização e limpeza.    |
| Análise Bivariada                    | Investigamos as relações entre variáveis e o impacto direto sobre o target (bom ou mau pagador), utilizando gráficos e correlações estatísticas. Essa visão comparativa ajudou a identificar os atributos com maior poder explicativo. |
| Correlação entre Atributos           | Geramos uma matriz de correlação para avaliar multicolinearidades e redundâncias entre variáveis, otimizando a base para modelagem futura e reduzindo ruído informacional.                                                               |
| Tratamento de Atributos Categóricos  | Variáveis qualitativas foram transformadas por meio de Label Encoding e *One-Hot Encoding, garantindo compatibilidade com algoritmos de machine learning.                                                                           |
| Balanceamento de Classes             | Aplicamos técnicas de oversampling e undersampling (via `imbalanced-learn`) para corrigir o desbalanceamento entre bons e maus pagadores — passo essencial para evitar viés nos modelos futuros.                                     |
| Divisão em Base de Treino e Teste    | Finalizamos a preparação dividindo o dataset em bases de treino (80%) e teste (20%), estruturando o pipeline para as próximas fases de modelagem preditiva.                                                                      |
````
</details>

### 🧾 Dicionário de Dados  
```markdown
| Variável              | Descrição                                                                 |
|-----------------------|---------------------------------------------------------------------------|
| **Age**               | Idade do cliente                                                         |
| **Income**            | Renda mensal                                                             |
| **Gender**            | Gênero do cliente                                                        |
| **Education**         | Nível de escolaridade                                                    |
| **Marital**           | Estado civil                                                             |
| **Number of Children**| Quantidade de filhos                                                     |
| **Home**              | Tipo de residência (alugada ou própria)                                  |
| **Credit Score**      | Score de crédito (variável-alvo)                                         |

````
---

## Etapa 1 - Pré Processamento  

### 🔹 Ações Realizadas  
- Verificação de **tipos de dados** e conversões necessárias.  
- Tratamento de **valores nulos e inconsistentes**, com justificativas documentadas.  
- Identificação e correção de **valores categóricos incorretos**.  

### 💡 Observação  
Foi aplicada **normalização** na variável *Income* (Renda), utilizando `MinMaxScaler`, para adequação à modelagem futura.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df["Income_Scaled"] = scaler.fit_transform(df[["Income"]])

```
## 📊 Etapa 2 - Análise Univariada e Bivariada

<details>
<summary><b>📈 Análise Univariada</b></summary>

### 🔸 Credit Score (Score de Crédito)

* A maioria dos clientes possui **score "High"**, indicando perfil de baixo risco.
* Scores “Average” e “Low” representam menor parcela, exigindo atenção especial na modelagem.

### 🔸 Age (Idade)

* Distribuição simétrica entre **28 e 45 anos**, mediana ≈ 36.
* Sem outliers significativos.

### 🔸 Home Ownership (Tipo de Moradia)

* Predominância de **casas próprias**, reforçando estabilidade financeira.

### 🔸 Income (Renda)

* Distribuição **enviesada à direita (skewed right)**.
* Renda concentrada entre **40k e 100k**, com cauda longa de altos rendimentos.
* Recomendação: **normalização ou transformação logarítmica** para uso em modelos ML.

</details>

---

<details>
<summary><b>🔗 Análise Bivariada</b></summary>

| Pergunta                                           | Insight                                            |
| -------------------------------------------------- | -------------------------------------------------- |
| **Existe relação entre a idade e o status civil?** | Sim. Clientes casados tendem a ser mais velhos.    |
| **Qual a relação entre score e escolaridade?**     | Maior escolaridade → score mais alto.              |
| **O salário influencia no score de crédito?**      | Renda maior → tendência a score “High”.            |
| **Clientes com casa própria têm score mais alto?** | Sim. 98,2% dos proprietários possuem score “High”. |



</details>


### 🧠 Conclusão da Etapa

> A estabilidade financeira e doméstica (renda e moradia própria) são os **principais preditores de baixo risco**.

---

## 📈 Etapa 3 - Correlação, Balanceamento e Codificação

### 🔹 Correlação Numérica

A relação entre **Age** e **Income** apresentou correlação média-alta (≈ 0.69).

> 💬 Justificativa: o aumento da idade reflete progressão profissional e aumento da renda — padrão esperado em bases financeiras.

### 🔹 Codificação Categórica

* **One-Hot Encoding:** Gender, Home Ownership, Marital Status
* **Label Encoding:** Education

### 🔹 Balanceamento das Classes

A variável *Credit Score* estava **desbalanceada**:

* “Average” → ~70%
* “Low” → ~20%
* “High” → ~10%

Foi aplicado **SMOTE** apenas na base de treino para equilibrar as classes.

```python
from imblearn.over_sampling import SMOTE

X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
print(Counter(y_res))
```

> 🧠 Resultado: melhor distribuição entre classes, reduzindo viés do modelo e garantindo aprendizado equilibrado.

---

## 🧭 Próximos Passos (Parte 2)

🔹 Construir e treinar modelos de classificação supervisionada:

* Logistic Regression
* Random Forest
* XGBoost

🔹 Avaliar métricas:

* Accuracy, Precision, Recall e F1-score
* Matriz de confusão
* AUC-ROC

🔹 Interpretar a importância das variáveis e gerar **insights preditivos** sobre o comportamento dos clientes.

---

## 💭 Reflexão Final

> O projeto demonstrou que **estabilidade financeira e social** (moradia própria, renda alta e escolaridade) são fatores decisivos na credibilidade de crédito.
> Essa compreensão é essencial para bancos e fintechs que desejam otimizar decisões de concessão de crédito.

---

## Autor


<p align="center">
  <b>Johnny Sorato Martins Fernandes</b><br>
  <sub>Consultoria de Negócios | Cientista de Dados| Analista de Dados - Automação de Processos - SaaS</sub><br><br>
  <sub> JS Fernandes Consultoria Empresarial - Unidade Primavera do Leste</sub><br><br>
  📧 fernandesjohnnys@gmail.com &nbsp;&nbsp;📞 (66) 99232-1719
</p>

---

## 📦 Instalação dos Requisitos

```bash
pip install -r requirements.txt
```

---

<p align="center">
  <i>“Dados bem tratados contam histórias, revelam padrões e constroem decisões inteligentes.”</i>
</p>
```

---

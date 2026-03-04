# tcc-final

INVESTIGAÇÃO E ANÁLISE DE AGRUPAMENTOS DE RÓTULOS CORRELACIONADOS A PARTIR DE MAPAS AUTO-ORGANIZÁVEIS DE KOHONEN (SOM)

## 📖 Como Usar

O projeto possui 2 scripts principais em `scripts/`:

### 1️⃣ **01_gridsearch.qmd** — Busca de Hiperparâmetros

**Função:** Testa diferentes combinações de parâmetros do SOM e encontra os melhores.

**Execução:**

```bash
# No RStudio: Ctrl+Shift+K (Knit/Render)
# Ou via terminal R:
quarto render scripts/01_gridsearch.qmd
```

**Saída:**

- `outputs/gridsearch/gridsearch_seed1234.csv` — Todos os resultados testados
- `outputs/gridsearch/best_params.csv` — Melhores parâmetros por dataset e vizinhança

**Tempo:** ~2-5 minutos

---

### 2️⃣ **02_evaluation_analysis.qmd** — Avaliação e Análise

**Função:** Treina o SOM com os melhores parâmetros encontrados e realiza análise detalhada.

**Execução (DEPOIS de rodar o script 1):**

```bash
# No RStudio: Ctrl+Shift+K
# Ou via terminal R:
quarto render scripts/02_evaluation_analysis.qmd
```

**Saída:**

- `outputs/evaluation/evaluation_results.csv` — Métricas finais
- `outputs/evaluation/*_label_freq.csv` — Frequências de rótulos
- `outputs/evaluation/*_labelset_freq.csv` — Frequências de labelsets

**Tempo:** ~2-5 minutos

---

## 🔧 Linhas que Devem Ser Alteradas

### Script 1: `scripts/01_gridsearch.qmd`

#### **Linhas 14-18** — Caminhos e Seed

```r
BASE_PATH      <- "E:/git-tcc/tcc-final/data"      # ← Altere se dados estão em outro local
OUTPUT_PATH    <- "E:/git-tcc/tcc-final/outputs/gridsearch"  # ← Altere se desejar salvar em outro local
SEED           <- 1234L                            # ← Altere para reproduzir com diferentes divisões
TARGET_DATASETS <- c("emotions", "scene")         # ← Adicione/remova datasets conforme necessário
```

**Quando alterar:**

- `BASE_PATH`: se os dados estão em outro diretório
- `SEED`: para gerar diferentes divisões aleatórias
- `TARGET_DATASETS`: para adicionar/remover datasets

---

#### **Linhas 27-38** — Configuração de Datasets

```r
datasets_config <- list(
    emotions = list(
        attr_cols   = 1:72,      # ← Altere se número de atributos for diferente
        label_cols  = 73:78,     # ← Altere se número de rótulos for diferente
        label_names = c("amazed.suprised", "happy.pleased", ...)  # ← Altere os nomes dos rótulos
    ),
    scene = list(
        attr_cols   = 1:294,     # ← Altere se número de atributos for diferente
        label_cols  = 295:300,   # ← Altere se número de rótulos for diferente
        label_names = c("Beach", "Sunset", ...)  # ← Altere os nomes dos rótulos
    )
)
```

**Quando alterar:**

- Se os atributos e rótulos estão em colunas diferentes
- Se os nomes dos rótulos mudam

---

#### **Linhas 97-100** — Parâmetros do Grid Search

```r
param_grid <- expand.grid(
    rlen        = c(100L, 200L, 500L),   # ← Altere para testar outros valores de iterações
    radius      = c(0.5, 1.0, 1.5),      # ← Altere para testar outros raios de vizinhança
    alpha_start = c(0.05),               # ← Altere taxa de aprendizado inicial
    alpha_end   = c(0.01),               # ← Altere taxa de aprendizado final
    ...
)
```

**Quando alterar:** Se quiser testar outros valores ou fazer grid search mais rápido/completo

---

### Script 2: `scripts/02_evaluation_analysis.qmd`

#### **Linhas 16-21** — Caminhos e Seed

```r
GRIDSEARCH_PATH     <- "E:/git-tcc/tcc-final/outputs/gridsearch"  # ← Altere se grid search está em outro local
BASE_PATH           <- "E:/git-tcc/tcc-final/data"               # ← Altere se dados estão em outro local
OUTPUT_PATH         <- "E:/git-tcc/tcc-final/outputs/evaluation" # ← Altere se desejar salvar em outro local
SEED                <- 1234L                                    # ← Altere para usar mesma seed do script 1
TARGET_DATASETS     <- c("emotions", "scene")                   # ← Deve ser IDÊNTICO ao script 1
TARGET_NEIGHBORHOODS <- c("gaussian", "bubble")                 # ← Altere se apenas um tipo interessa
```

**⚠️ IMPORTANTE:**

- `SEED` deve ser igual ao do script 1
- `TARGET_DATASETS` deve ser igual ao do script 1
- `BASE_PATH` e `GRIDSEARCH_PATH` devem estar corretos

---

#### **Linhas 28-39** — Configuração de Datasets

```r
datasets_config <- list(
    emotions = list(
        attr_cols   = 1:72,
        label_cols  = 73:78,
        label_names = c("amazed.suprised", "happy.pleased", ...)
    ),
    scene = list(
        attr_cols   = 1:294,
        label_cols  = 295:300,
        label_names = c("Beach", "Sunset", ...)
    )
)
```

**⚠️ IMPORTANTE:** Deve ser **IDÊNTICO** ao do script 1

---

#### **Linha 60** — Escolher Critério de "Melhores Parâmetros"

```r
# best_file <- file.path(GRIDSEARCH_PATH, "best_params_min.csv")  # ← Descomente para usar mínimo
best_file <- file.path(GRIDSEARCH_PATH, "best_params.csv")       # ← Usa média (padrão)
```

**Quando alterar:**

- Use `best_params.csv` para resulados mais **estáveis** (média dos 3 folds)
- Use `best_params_min.csv` para resultados mais **agressivos** (mínimo)

---

## 📁 Estrutura de Dados

Os scripts esperam a seguinte estrutura em `data/`:

```
data/
├── emotions/
│   └── Stratified/CrossValidation/
│       ├── Tr/  (dados de treino, 3 folds)
│       ├── Vl/  (dados de validação, 3 folds)
│       └── Ts/  (dados de teste, 3 folds)
└── scene/
    └── Stratified/CrossValidation/
        ├── Tr/
        ├── Vl/
        └── Ts/
```

**Tipo de arquivo esperado:**

- `emotions-Split-Tr-1.csv`, `emotions-Split-Vl-1.csv`, `emotions-Split-Ts-1.csv`, etc.

---

## 📋 Checklist Antes de Executar

- [ ] R com pacotes instalados: `tidyverse`, `kohonen`, `gridExtra`
- [ ] Pasta `data/emotions/Stratified/CrossValidation/` com datasets
- [ ] Pasta `data/scene/Stratified/CrossValidation/` com datasets
- [ ] Verificar se `BASE_PATH` está correto nos scripts
- [ ] **Executar Script 1 primeiro**, depois Script 2

---

## 🎯 Resumo Rápido

| Ação                          | Comando                                              | Tempo    |
| ------------------------------- | ---------------------------------------------------- | -------- |
| Buscar melhores parâmetros     | `quarto render scripts/01_gridsearch.qmd`          | 2-5 min |
| Avaliar modelo                  | `quarto render scripts/02_evaluation_analysis.qmd` | 2-5 min  |
| Apenas renderizar para markdown | Adicione `format: markdown` no `.qmd`            | Rápido  |
| Apenas renderizar para PDF      | Use `format: pdf` (padrão do script 2)            | Rápido  |

# Script 2 — Avaliação do Modelo e Análise de Partições


## Configuração

``` r
library(tidyverse)
```

    ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ✔ dplyr     1.1.4     ✔ readr     2.1.6
    ✔ forcats   1.0.1     ✔ stringr   1.6.0
    ✔ ggplot2   4.0.1     ✔ tibble    3.3.1
    ✔ lubridate 1.9.4     ✔ tidyr     1.3.2
    ✔ purrr     1.2.1     
    ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ✖ dplyr::filter() masks stats::filter()
    ✖ dplyr::lag()    masks stats::lag()
    ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(kohonen)
```


    Attaching package: 'kohonen'

    The following object is masked from 'package:purrr':

        map

``` r
library(gridExtra)
```


    Attaching package: 'gridExtra'

    The following object is masked from 'package:dplyr':

        combine

``` r
library(grid)
library(cluster)      # silhouette, agnes

GRIDSEARCH_PATH      <- "E:/git-tcc/tcc-final/outputs/gridsearch"
BASE_PATH            <- "E:/git-tcc/tcc-final/data"
OUTPUT_PATH          <- "E:/git-tcc/tcc-final/outputs/evaluation"
SEED                 <- 1234L
TARGET_DATASETS      <- c("emotions", "scene")
TARGET_NEIGHBORHOODS <- c("gaussian", "bubble")

dir.create(OUTPUT_PATH, recursive = TRUE, showWarnings = FALSE)
set.seed(SEED)
```

## Configuração dos Datasets

``` r
datasets_config <- list(
    emotions = list(
        attr_cols   = 1:72,
        label_cols  = 73:78,
        label_names = c("amazed.suprised", "happy.pleased", "relaxing.calm",
                        "quiet.still", "sad.lonely", "angry.aggresive")
    ),
    scene = list(
        attr_cols   = 1:294,
        label_cols  = 295:300,
        label_names = c("Beach", "Sunset", "FallFoliage", "Field", "Mountain", "Urban")
    )
)
```

## Carregamento dos Dados e Melhores Parâmetros

``` r
load_fold <- function(base_path, dataset_name, config, fold, types = c("Tr", "Ts")) {
    result <- list()
    for (type in types) {
        path <- file.path(
            base_path, dataset_name, "Stratified", "CrossValidation", type,
            sprintf("%s-Split-%s-%d.csv", dataset_name, type, fold)
        )
        raw <- read.csv(path)
        result[[tolower(type)]] <- list(
            attr   = raw[, config$attr_cols],
            labels = raw[, config$label_cols]
        )
    }
    result
}

datasets <- list()
for (ds in TARGET_DATASETS) {
    datasets[[ds]] <- list(config = datasets_config[[ds]])
    for (fold in 1:3) {
        datasets[[ds]][[paste0("fold", fold)]] <-
            load_fold(BASE_PATH, ds, datasets_config[[ds]], fold)
    }
}

# best_file <- file.path(GRIDSEARCH_PATH, "best_params.csv")
best_file <- file.path(GRIDSEARCH_PATH, "best_params_min.csv")
if (!file.exists(best_file)) {
    stop(sprintf(
        "Arquivo de melhores parametros nao encontrado: %s\nExecute o script 01_gridsearch.qmd primeiro.",
        best_file
    ))
}

best_params <- read.csv(best_file, stringsAsFactors = FALSE)

cat("Melhores parametros carregados:\n")
```

    Melhores parametros carregados:

``` r
print(best_params)
```

       dataset neighborhood    topology rlen radius alpha_start alpha_end    min_qe
    1 emotions       bubble rectangular  100      2        0.05      0.01 0.4586958
    2 emotions     gaussian   hexagonal  100      2        0.05      0.01 0.5421067
    3    scene       bubble rectangular 1000      1        0.05      0.01 0.3586678
    4    scene     gaussian   hexagonal  500      1        0.05      0.01 0.4261595
        mean_qe       sd_qe n_folds
    1 0.4683699 0.012988941       3
    2 0.5485781 0.007372719       3
    3 0.3653513 0.006523930       3
    4 0.4355491 0.009735314       3

## Funções Auxiliares

``` r
# Treina um supersom com os parâmetros fornecidos
train_som <- function(Y_train, topology, neighborhood, rlen, radius, alpha_start, alpha_end) {
    grid <- kohonen::somgrid(2, 2, topology, neighbourhood.fct = neighborhood)
    kohonen::supersom(
        data      = list(Y = Y_train),
        grid      = grid,
        rlen      = rlen,
        radius    = radius,
        alpha     = c(alpha_start, alpha_end),
        keep.data = TRUE
    )
}

# Aplica clustering hierárquico sobre os codebook vectors
# Retorna o dendrograma e o vetor de clusters (posição i = cluster do neurônio i)
cluster_codebook <- function(model, k, method = "complete") {
    codebook <- model$codes[[1]]
    dend     <- hclust(dist(codebook), method = method)
    clusters <- as.integer(cutree(dend, k = k))
    list(dend = dend, clusters = clusters)
}

# Constrói dataframe: instância → neurônio, cluster e rótulos
build_instance_df <- function(model, Y_train, label_names, clusters) {
    df <- data.frame(
        neuron  = model$unit.classif,
        cluster = clusters[model$unit.classif],
        Y_train
    )
    colnames(df) <- c("neuron", "cluster", label_names)
    df
}

# Frequência individual dos rótulos por cluster
label_freq_by_cluster <- function(instance_df, label_names) {
    instance_df |>
        dplyr::group_by(cluster) |>
        dplyr::summarise(
            n_instancias = dplyr::n(),
            dplyr::across(dplyr::all_of(label_names), sum),
            .groups = "drop"
        )
}

# Frequência dos labelsets (combinação binária de rótulos) por cluster
labelset_freq_by_cluster <- function(instance_df, label_names) {
    instance_df |>
        dplyr::mutate(
            labelset = apply(dplyr::pick(dplyr::all_of(label_names)), 1, paste, collapse = "")
        ) |>
        dplyr::count(cluster, labelset, name = "frequencia") |>
        dplyr::arrange(cluster, dplyr::desc(frequencia))
}

# Renderiza um dataframe como tabela numa nova página do PDF
render_table_page <- function(df, title = NULL) {
    grid::grid.newpage()
    if (!is.null(title)) {
        grid::pushViewport(grid::viewport(
            layout = grid::grid.layout(2, 1, heights = grid::unit(c(2, 1), c("lines", "null")))
        ))
        grid::pushViewport(grid::viewport(layout.pos.row = 1))
        grid::grid.text(title, gp = grid::gpar(fontsize = 12, fontface = "bold"))
        grid::popViewport()
        grid::pushViewport(grid::viewport(layout.pos.row = 2))
        gridExtra::grid.table(df, rows = NULL)
        grid::popViewport()
        grid::popViewport()
    } else {
        gridExtra::grid.table(df, rows = NULL)
    }
}
```

## Funções de Qualidade de Clustering

``` r
# 1. Correlação Cofenética (CCC) — antes do corte
# Mede o quão bem o dendrograma preserva as distâncias originais entre codebook vectors.
# Valores > 0.75 indicam boa representação. Usado para comparar linkages e validar
# que a estrutura hierárquica encontrada é fiel à geometria dos neurônios.
compute_ccc <- function(codebook, dend) {
    cor(dist(codebook), cophenetic(dend))
}

# 2. Coeficiente Aglomerativo (AC) — antes do corte
# Mede o grau de estrutura hierárquica nos dados: quão "limpas" são as fusões.
# Complementa o CCC — CCC avalia fidelidade às distâncias, AC avalia clareza da hierarquia.
# Valores > 0.75 indicam estrutura hierárquica bem definida.
compute_ac <- function(codebook, method = "complete") {
    cluster::agnes(dist(codebook), method = method)$ac
}

# 3. Silhouette médio — após o corte
# Para cada neurônio: quão similar ele é ao próprio cluster vs. o cluster vizinho.
# Valores próximos de +1 indicam boa alocação; negativos indicam alocação errada.
# Métrica mais consolidada na literatura para avaliar qualidade de partições.
compute_silhouette <- function(clusters, codebook) {
    if (length(unique(clusters)) < 2) return(NA_real_)
    sil <- cluster::silhouette(clusters, dist(codebook))
    mean(sil[, "sil_width"])
}

# 4. Entropia média dos labelsets por cluster — após o corte
# Clusters com entropia baixa concentram poucos labelsets dominantes,
# indicando boa separação semântica no espaço multilabel.
# Retorna a média da entropia de Shannon entre todos os clusters.
compute_labelset_entropy <- function(lset_freq) {
    lset_freq |>
        dplyr::group_by(cluster) |>
        dplyr::summarise(
            entropy = {
                p <- frequencia / sum(frequencia)
                -sum(p * log2(p + 1e-10))
            },
            .groups = "drop"
        ) |>
        dplyr::summarise(mean_entropy = mean(entropy)) |>
        dplyr::pull(mean_entropy)
}

# 5. Distância de Hellinger entre distribuições de rótulos — após o corte, k=2
# Quantifica a separação semântica entre os dois clusters no espaço de rótulos.
# Varia de 0 (distribuições idênticas) a 1 (distribuições completamente distintas).
# Aplicável apenas para k=2; retorna NA para k>2.
compute_hellinger <- function(label_freq, label_names) {
    if (nrow(label_freq) != 2) return(NA_real_)
    p <- as.numeric(label_freq[1, label_names])
    q <- as.numeric(label_freq[2, label_names])
    p <- p / sum(p)
    q <- q / sum(q)
    sqrt(0.5 * sum((sqrt(p) - sqrt(q))^2))
}

# 6. Consistência entre folds — calculada após consolidar todos os folds
# Para cada (dataset, vizinhança, k, cluster), computa o desvio padrão da
# proporção de cada rótulo entre os 3 folds. SD baixo = partição estável e
# reproduzível — essencial para validar os resultados em cross-validation.
# Implementada como group_modify para respeitar os label_names de cada dataset.
compute_fold_consistency <- function(label_freq_consolidated, datasets_config) {
    label_freq_consolidated |>
        dplyr::group_by(dataset, neighborhood, k, cluster) |>
        dplyr::group_modify(function(group_df, keys) {
            ds_labels <- datasets_config[[keys$dataset]]$label_names
            group_df |>
                dplyr::mutate(
                    dplyr::across(dplyr::all_of(ds_labels),
                                  ~ .x / n_instancias,
                                  .names = "prop_{.col}")
                ) |>
                dplyr::summarise(
                    n_folds = dplyr::n(),
                    dplyr::across(
                        dplyr::starts_with("prop_"),
                        list(sd = ~ sd(.x, na.rm = TRUE)),
                        .names = "{.col}_{.fn}"
                    ),
                    mean_sd_props = mean(
                        dplyr::c_across(
                            dplyr::starts_with("prop_") & dplyr::ends_with("_sd")
                        ),
                        na.rm = TRUE
                    ),
                    .groups = "drop"
                )
        }) |>
        dplyr::ungroup()
}
```

## Treinamento e Avaliação nos 3 Folds

``` r
trained_models <- list()
eval_results   <- list()

for (ds in TARGET_DATASETS) {
    trained_models[[ds]] <- list()

    for (nbhd in TARGET_NEIGHBORHOODS) {
        trained_models[[ds]][[nbhd]] <- list()

        bp <- best_params |>
            dplyr::filter(dataset == ds, neighborhood == nbhd) |>
            head(1)

        if (nrow(bp) == 0) {
            warning(sprintf("Parametros nao encontrados para %s + %s. Pulando.", ds, nbhd))
            next
        }

        cat(sprintf(
            "\n=== %s | %s | topology=%s rlen=%d radius=%f alpha=(%.2f, %.3f) ===\n",
            toupper(ds), toupper(nbhd),
            bp$topology, bp$rlen, bp$radius, bp$alpha_start, bp$alpha_end
        ))

        for (fold in 1:3) {
            fold_key <- paste0("fold", fold)

            Y_tr <- as.matrix(datasets[[ds]][[fold_key]][["tr"]]$labels)
            Y_ts <- as.matrix(datasets[[ds]][[fold_key]][["ts"]]$labels)

            model <- tryCatch({
                train_som(Y_tr, bp$topology, nbhd, bp$rlen, bp$radius, bp$alpha_start, bp$alpha_end)
            }, error = function(e) {
                message("Erro no fold ", fold, ": ", e$message)
                NULL
            })

            if (is.null(model)) next

            grid_size     <- model$grid$xdim * model$grid$ydim
            empty_neurons <- grid_size - length(unique(model$unit.classif))
            qe_test       <- mean(kohonen::map(model, newdata = list(Y = Y_ts))$distances)

            cat(sprintf("  Fold %d: QE_test=%.5f | neuronios_vazios=%d\n",
                        fold, qe_test, empty_neurons))

            trained_models[[ds]][[nbhd]][[fold_key]] <- list(
                model         = model,
                Y_tr          = Y_tr,
                Y_ts          = Y_ts,
                qe_test       = qe_test,
                empty_neurons = empty_neurons
            )

            eval_results[[length(eval_results) + 1]] <- data.frame(
                dataset       = ds,
                neighborhood  = nbhd,
                topology      = bp$topology,
                fold          = fold,
                rlen          = bp$rlen,
                radius        = bp$radius,
                alpha_start   = bp$alpha_start,
                alpha_end     = bp$alpha_end,
                empty_neurons = empty_neurons,
                qe_test       = qe_test
            )
        }
    }
}
```


    === EMOTIONS | GAUSSIAN | topology=hexagonal rlen=100 radius=2.000000 alpha=(0.05, 0.010) ===
      Fold 1: QE_test=0.63653 | neuronios_vazios=0
      Fold 2: QE_test=0.57476 | neuronios_vazios=0
      Fold 3: QE_test=0.58248 | neuronios_vazios=0

    === EMOTIONS | BUBBLE | topology=rectangular rlen=100 radius=2.000000 alpha=(0.05, 0.010) ===
      Fold 1: QE_test=0.48051 | neuronios_vazios=0
      Fold 2: QE_test=0.51445 | neuronios_vazios=0
      Fold 3: QE_test=0.48217 | neuronios_vazios=0

    === SCENE | GAUSSIAN | topology=hexagonal rlen=500 radius=1.000000 alpha=(0.05, 0.010) ===
      Fold 1: QE_test=0.44081 | neuronios_vazios=0
      Fold 2: QE_test=0.43232 | neuronios_vazios=0
      Fold 3: QE_test=0.43573 | neuronios_vazios=0

    === SCENE | BUBBLE | topology=rectangular rlen=1000 radius=1.000000 alpha=(0.05, 0.010) ===
      Fold 1: QE_test=0.37580 | neuronios_vazios=0
      Fold 2: QE_test=0.37302 | neuronios_vazios=0
      Fold 3: QE_test=0.37686 | neuronios_vazios=0

``` r
eval_df <- dplyr::bind_rows(eval_results)
write.csv(eval_df, file.path(OUTPUT_PATH, "evaluation_results.csv"), row.names = FALSE)

cat("\n=== RESULTADOS DE AVALIACAO ===\n")
```


    === RESULTADOS DE AVALIACAO ===

``` r
print(eval_df)
```

        dataset neighborhood    topology fold rlen radius alpha_start alpha_end
    1  emotions     gaussian   hexagonal    1  100      2        0.05      0.01
    2  emotions     gaussian   hexagonal    2  100      2        0.05      0.01
    3  emotions     gaussian   hexagonal    3  100      2        0.05      0.01
    4  emotions       bubble rectangular    1  100      2        0.05      0.01
    5  emotions       bubble rectangular    2  100      2        0.05      0.01
    6  emotions       bubble rectangular    3  100      2        0.05      0.01
    7     scene     gaussian   hexagonal    1  500      1        0.05      0.01
    8     scene     gaussian   hexagonal    2  500      1        0.05      0.01
    9     scene     gaussian   hexagonal    3  500      1        0.05      0.01
    10    scene       bubble rectangular    1 1000      1        0.05      0.01
    11    scene       bubble rectangular    2 1000      1        0.05      0.01
    12    scene       bubble rectangular    3 1000      1        0.05      0.01
       empty_neurons   qe_test
    1              0 0.6365260
    2              0 0.5747556
    3              0 0.5824840
    4              0 0.4805075
    5              0 0.5144453
    6              0 0.4821744
    7              0 0.4408123
    8              0 0.4323155
    9              0 0.4357299
    10             0 0.3758018
    11             0 0.3730216
    12             0 0.3768609

## Geração de Partições e Métricas de Qualidade

``` r
# Para cada (dataset, vizinhança, fold):
#   - Antes do corte : CCC e AC sobre o dendrograma
#   - Após o corte   : Silhouette, Entropia dos labelsets e Hellinger (k=2)
#   - Entre folds    : Consistência (SD das proporções de rótulos) — calculada ao final

partition2_data       <- list()
hierarchy_results     <- list()   # CCC e AC por fold (pré-corte)
partition_quality_res <- list()   # Silhouette, Entropia, Hellinger por (fold, k)
label_freq_all_folds  <- list()   # acumula label_freq de todos os combos/folds/ks

for (ds in TARGET_DATASETS) {
    label_names <- datasets_config[[ds]]$label_names

    for (nbhd in TARGET_NEIGHBORHOODS) {
        combo <- sprintf("%s_%s", ds, nbhd)
        label_freq_folds    <- list()
        labelset_freq_folds <- list()

        for (fold in 1:3) {
            fold_key   <- paste0("fold", fold)
            model_info <- trained_models[[ds]][[nbhd]][[fold_key]]
            if (is.null(model_info)) next

            model  <- model_info$model
            Y_tr   <- model_info$Y_tr
            n_neur <- model$grid$xdim * model$grid$ydim
            max_k  <- n_neur - 1

            codebook <- model$codes[[1]]
            dend     <- hclust(dist(codebook), method = "complete")

            # --- Métricas pré-corte: CCC e AC ---
            ccc_val <- compute_ccc(codebook, dend)
            ac_val  <- compute_ac(codebook, method = "complete")

            hierarchy_results[[length(hierarchy_results) + 1]] <- data.frame(
                dataset      = ds,
                neighborhood = nbhd,
                fold         = fold,
                ccc          = ccc_val,
                ac           = ac_val
            )

            cat(sprintf("  [%s | %s | Fold %d] CCC=%.4f | AC=%.4f\n",
                        ds, nbhd, fold, ccc_val, ac_val))

            pdf_path <- file.path(OUTPUT_PATH, sprintf("%s_fold%d_partitions.pdf", combo, fold))
            pdf(pdf_path, width = 11, height = 8.5)

            # --- Visão geral do modelo ---
            par(mfrow = c(2, 2))
            plot(model, type = "counts",
                 main = sprintf("%s | %s | Fold %d - Counts", ds, nbhd, fold))
            plot(model, type = "changes",
                 main = "Convergencia do treinamento")
            plot(model, type = "dist.neighbours",
                 main = "U-Matrix (distancia entre vizinhos)")
            plot(model, type = "codes", codeRendering = "segments",
                 main = "Codebook - Segments")
            par(mfrow = c(1, 1))

            # --- Partições k = 2 até max_k ---
            for (k in 2:max_k) {
                clusters    <- as.integer(cutree(dend, k = k))
                instance_df <- build_instance_df(model, Y_tr, label_names, clusters)
                label_freq  <- label_freq_by_cluster(instance_df, label_names)
                lset_freq   <- labelset_freq_by_cluster(instance_df, label_names)

                # --- Métricas pós-corte ---
                sil_val       <- compute_silhouette(clusters, codebook)
                entropy_val   <- compute_labelset_entropy(lset_freq)
                hellinger_val <- compute_hellinger(label_freq, label_names)

                partition_quality_res[[length(partition_quality_res) + 1]] <- data.frame(
                    dataset      = ds,
                    neighborhood = nbhd,
                    fold         = fold,
                    k            = k,
                    silhouette   = sil_val,
                    mean_entropy = entropy_val,
                    hellinger    = hellinger_val
                )

                cat(sprintf(
                    "    k=%d: Silhouette=%.4f | Entropia=%.4f | Hellinger=%s\n",
                    k, sil_val, entropy_val,
                    ifelse(is.na(hellinger_val), "NA", sprintf("%.4f", hellinger_val))
                ))

                # Acumula para consistência entre folds (métrica 6)
                label_freq_all_folds[[length(label_freq_all_folds) + 1]] <- label_freq |>
                    dplyr::mutate(dataset = ds, neighborhood = nbhd, fold = fold, k = k)

                # --- Visualizações no PDF ---
                plot(model, type = "codes", codeRendering = "segments",
                     bgcol = rainbow(k)[clusters],
                     main  = sprintf("Clusters (k=%d) | %s | %s | Fold %d", k, ds, nbhd, fold))
                kohonen::add.cluster.boundaries(model, clusters)
                pts <- model$grid$pts
                text(pts[, 1], pts[, 2],
                     labels = sprintf("N%d\nC%d", seq_len(nrow(pts)), clusters),
                     cex = 0.75, font = 2)

                plot(dend,
                     main = sprintf("Dendrograma (k=%d) | %s | %s | Fold %d", k, ds, nbhd, fold),
                     xlab = "Neuronios", ylab = "Distancia")
                rect.hclust(dend, k = k, border = "red")

                # Tabela: métricas de qualidade desta partição no PDF
                quality_row <- data.frame(
                    Metrica = c("Silhouette medio", "Entropia media dos labelsets",
                                "Hellinger (rotulos)", "CCC (pre-corte)", "AC (pre-corte)"),
                    Valor   = c(
                        sprintf("%.4f", sil_val),
                        sprintf("%.4f", entropy_val),
                        ifelse(is.na(hellinger_val), "NA (k>2)", sprintf("%.4f", hellinger_val)),
                        sprintf("%.4f", ccc_val),
                        sprintf("%.4f", ac_val)
                    )
                )
                render_table_page(
                    quality_row,
                    title = sprintf("Metricas de qualidade (k=%d) | %s | %s | Fold %d",
                                    k, ds, nbhd, fold)
                )

                render_table_page(
                    as.data.frame(label_freq),
                    title = sprintf("Frequencia de rotulos por cluster (k=%d) | %s | %s | Fold %d",
                                    k, ds, nbhd, fold)
                )

                for (cl_id in sort(unique(lset_freq$cluster))) {
                    top_ls <- lset_freq |>
                        dplyr::filter(cluster == cl_id) |>
                        head(15)
                    render_table_page(
                        as.data.frame(top_ls),
                        title = sprintf("Top 15 labelsets — Cluster %d (k=%d) | %s | %s | Fold %d",
                                        cl_id, k, ds, nbhd, fold)
                    )
                }

                if (k == 2) {
                    label_freq_folds[[fold_key]] <- label_freq |>
                        dplyr::mutate(fold = fold, dataset = ds, neighborhood = nbhd)
                    labelset_freq_folds[[fold_key]] <- lset_freq |>
                        dplyr::mutate(fold = fold, dataset = ds, neighborhood = nbhd)
                }
            }

            dev.off()
            cat(sprintf("  PDF gerado: %s\n", basename(pdf_path)))
        }

        if (length(label_freq_folds) == 0) next

        lf_p2  <- dplyr::bind_rows(label_freq_folds)
        lsf_p2 <- dplyr::bind_rows(labelset_freq_folds)

        write.csv(lf_p2,
                  file.path(OUTPUT_PATH, sprintf("%s_p2_label_freq.csv", combo)),
                  row.names = FALSE)
        write.csv(lsf_p2,
                  file.path(OUTPUT_PATH, sprintf("%s_p2_labelset_freq.csv", combo)),
                  row.names = FALSE)

        partition2_data[[combo]] <- list(
            label_freq    = lf_p2,
            labelset_freq = lsf_p2
        )

        cat(sprintf("  CSVs da particao 2 salvos para: %s\n", combo))
    }
}
```

      [emotions | gaussian | Fold 1] CCC=0.6777 | AC=0.3665

        k=2: Silhouette=0.2029 | Entropia=2.6764 | Hellinger=0.7344

        k=3: Silhouette=0.0885 | Entropia=2.3520 | Hellinger=NA

      PDF gerado: emotions_gaussian_fold1_partitions.pdf
      [emotions | gaussian | Fold 2] CCC=0.5147 | AC=0.3856

        k=2: Silhouette=0.1693 | Entropia=2.9333 | Hellinger=0.7167

        k=3: Silhouette=0.1087 | Entropia=2.2754 | Hellinger=NA

      PDF gerado: emotions_gaussian_fold2_partitions.pdf
      [emotions | gaussian | Fold 3] CCC=0.5400 | AC=0.3425

        k=2: Silhouette=0.1416 | Entropia=2.7152 | Hellinger=0.7513

        k=3: Silhouette=0.0913 | Entropia=2.2772 | Hellinger=NA

      PDF gerado: emotions_gaussian_fold3_partitions.pdf
      CSVs da particao 2 salvos para: emotions_gaussian
      [emotions | bubble | Fold 1] CCC=0.6603 | AC=0.2622

        k=2: Silhouette=0.1484 | Entropia=2.8955 | Hellinger=0.7141

        k=3: Silhouette=0.1068 | Entropia=2.2876 | Hellinger=NA

      PDF gerado: emotions_bubble_fold1_partitions.pdf
      [emotions | bubble | Fold 2] CCC=0.6429 | AC=0.3012

        k=2: Silhouette=0.1533 | Entropia=2.9263 | Hellinger=0.7069

        k=3: Silhouette=0.0974 | Entropia=2.3209 | Hellinger=NA

      PDF gerado: emotions_bubble_fold2_partitions.pdf
      [emotions | bubble | Fold 3] CCC=0.5845 | AC=0.2459

        k=2: Silhouette=0.1229 | Entropia=2.9056 | Hellinger=0.7254

        k=3: Silhouette=0.0943 | Entropia=2.3058 | Hellinger=NA

      PDF gerado: emotions_bubble_fold3_partitions.pdf
      CSVs da particao 2 salvos para: emotions_bubble
      [scene | gaussian | Fold 1] CCC=0.8749 | AC=0.1690

        k=2: Silhouette=0.1119 | Entropia=1.8017 | Hellinger=0.8262

        k=3: Silhouette=0.0188 | Entropia=1.2532 | Hellinger=NA

      PDF gerado: scene_gaussian_fold1_partitions.pdf
      [scene | gaussian | Fold 2] CCC=0.6119 | AC=0.2318

        k=2: Silhouette=0.0916 | Entropia=1.7929 | Hellinger=0.8043

        k=3: Silhouette=0.0735 | Entropia=1.2021 | Hellinger=NA

      PDF gerado: scene_gaussian_fold2_partitions.pdf
      [scene | gaussian | Fold 3] CCC=0.4748 | AC=0.1395

        k=2: Silhouette=0.0516 | Entropia=1.9310 | Hellinger=0.9253

        k=3: Silhouette=0.0219 | Entropia=1.3362 | Hellinger=NA

      PDF gerado: scene_gaussian_fold3_partitions.pdf
      CSVs da particao 2 salvos para: scene_gaussian
      [scene | bubble | Fold 1] CCC=0.8306 | AC=0.1719

        k=2: Silhouette=0.0845 | Entropia=1.3689 | Hellinger=1.0000

        k=3: Silhouette=0.0774 | Entropia=1.0172 | Hellinger=NA

      PDF gerado: scene_bubble_fold1_partitions.pdf
      [scene | bubble | Fold 2] CCC=0.5297 | AC=0.1056

        k=2: Silhouette=0.0583 | Entropia=1.3699 | Hellinger=1.0000

        k=3: Silhouette=0.0499 | Entropia=0.9801 | Hellinger=NA

      PDF gerado: scene_bubble_fold2_partitions.pdf
      [scene | bubble | Fold 3] CCC=0.5632 | AC=0.1124

        k=2: Silhouette=0.0616 | Entropia=1.3688 | Hellinger=1.0000

        k=3: Silhouette=0.0542 | Entropia=1.0242 | Hellinger=NA

      PDF gerado: scene_bubble_fold3_partitions.pdf
      CSVs da particao 2 salvos para: scene_bubble

``` r
# --- Consolida e salva métricas pré e pós corte ---
hierarchy_df      <- dplyr::bind_rows(hierarchy_results)
partition_qual_df <- dplyr::bind_rows(partition_quality_res)

write.csv(hierarchy_df,
          file.path(OUTPUT_PATH, "quality_hierarchy.csv"),
          row.names = FALSE)
write.csv(partition_qual_df,
          file.path(OUTPUT_PATH, "quality_partitions.csv"),
          row.names = FALSE)

cat("\n=== METRICAS PRE-CORTE (CCC e AC) ===\n")
```


    === METRICAS PRE-CORTE (CCC e AC) ===

``` r
print(hierarchy_df)
```

        dataset neighborhood fold       ccc        ac
    1  emotions     gaussian    1 0.6776890 0.3664746
    2  emotions     gaussian    2 0.5146758 0.3856457
    3  emotions     gaussian    3 0.5400375 0.3424901
    4  emotions       bubble    1 0.6602701 0.2622266
    5  emotions       bubble    2 0.6429348 0.3012106
    6  emotions       bubble    3 0.5845408 0.2458527
    7     scene     gaussian    1 0.8748631 0.1690285
    8     scene     gaussian    2 0.6118725 0.2318480
    9     scene     gaussian    3 0.4747806 0.1394791
    10    scene       bubble    1 0.8306285 0.1718968
    11    scene       bubble    2 0.5297298 0.1056313
    12    scene       bubble    3 0.5631609 0.1124128

``` r
cat("\n=== METRICAS POS-CORTE (Silhouette, Entropia, Hellinger) ===\n")
```


    === METRICAS POS-CORTE (Silhouette, Entropia, Hellinger) ===

``` r
print(partition_qual_df)
```

        dataset neighborhood fold k silhouette mean_entropy hellinger
    1  emotions     gaussian    1 2 0.20291673    2.6764447 0.7343793
    2  emotions     gaussian    1 3 0.08851036    2.3519673        NA
    3  emotions     gaussian    2 2 0.16930618    2.9333153 0.7167256
    4  emotions     gaussian    2 3 0.10871400    2.2754351        NA
    5  emotions     gaussian    3 2 0.14160334    2.7152078 0.7512889
    6  emotions     gaussian    3 3 0.09130938    2.2771548        NA
    7  emotions       bubble    1 2 0.14844895    2.8954653 0.7141262
    8  emotions       bubble    1 3 0.10680158    2.2876104        NA
    9  emotions       bubble    2 2 0.15333362    2.9262850 0.7068771
    10 emotions       bubble    2 3 0.09736993    2.3209222        NA
    11 emotions       bubble    3 2 0.12290569    2.9055700 0.7253743
    12 emotions       bubble    3 3 0.09425780    2.3057535        NA
    13    scene     gaussian    1 2 0.11190522    1.8017299 0.8261867
    14    scene     gaussian    1 3 0.01882860    1.2532139        NA
    15    scene     gaussian    2 2 0.09162338    1.7928642 0.8043213
    16    scene     gaussian    2 3 0.07347641    1.2021272        NA
    17    scene     gaussian    3 2 0.05161049    1.9309539 0.9252715
    18    scene     gaussian    3 3 0.02191673    1.3362067        NA
    19    scene       bubble    1 2 0.08453167    1.3689467 1.0000000
    20    scene       bubble    1 3 0.07735344    1.0172285        NA
    21    scene       bubble    2 2 0.05825441    1.3698790 1.0000000
    22    scene       bubble    2 3 0.04988482    0.9800795        NA
    23    scene       bubble    3 2 0.06161580    1.3687949 1.0000000
    24    scene       bubble    3 3 0.05419483    1.0241886        NA

## Consistência Entre Folds

``` r
# Métrica 6: SD das proporções de rótulos por cluster entre os 3 folds.
# SD baixo indica que a partição distribui os rótulos de forma estável e
# reproduzível entre os folds — essencial para confiar nos resultados
# de cross-validation.

label_freq_consolidated <- dplyr::bind_rows(label_freq_all_folds)

fold_consistency_df <- compute_fold_consistency(label_freq_consolidated, datasets_config)

write.csv(fold_consistency_df,
          file.path(OUTPUT_PATH, "quality_fold_consistency.csv"),
          row.names = FALSE)

cat("=== CONSISTENCIA ENTRE FOLDS (mean_sd_props — quanto menor, mais estavel) ===\n")
```

    === CONSISTENCIA ENTRE FOLDS (mean_sd_props — quanto menor, mais estavel) ===

``` r
fold_consistency_df |>
    dplyr::select(dataset, neighborhood, k, cluster, n_folds, mean_sd_props) |>
    print()
```

    # A tibble: 20 × 6
       dataset  neighborhood     k cluster n_folds mean_sd_props
       <chr>    <chr>        <int>   <int>   <int>         <dbl>
     1 emotions bubble           2       1       3      0.277   
     2 emotions bubble           2       2       3      0.300   
     3 emotions bubble           3       1       3      0.228   
     4 emotions bubble           3       2       3      0.416   
     5 emotions bubble           3       3       3      0.269   
     6 emotions gaussian         2       1       3      0.110   
     7 emotions gaussian         2       2       3      0.129   
     8 emotions gaussian         3       1       3      0.246   
     9 emotions gaussian         3       2       3      0.337   
    10 emotions gaussian         3       3       3      0.449   
    11 scene    bubble           2       1       3      0.000848
    12 scene    bubble           2       2       3      0       
    13 scene    bubble           3       1       3      0.0520  
    14 scene    bubble           3       2       3      0.307   
    15 scene    bubble           3       3       3      0.206   
    16 scene    gaussian         2       1       3      0.133   
    17 scene    gaussian         2       2       3      0.278   
    18 scene    gaussian         3       1       3      0.207   
    19 scene    gaussian         3       2       3      0.278   
    20 scene    gaussian         3       3       3      0.297   

## Sumário Visual das Métricas de Qualidade

``` r
# Gráfico 1: CCC e AC por (dataset, neighborhood, fold)
p_hierarchy <- hierarchy_df |>
    tidyr::pivot_longer(cols = c(ccc, ac), names_to = "metrica", values_to = "valor") |>
    dplyr::mutate(
        metrica = dplyr::recode(metrica,
                                ccc = "Correlacao Cofenotica (CCC)",
                                ac  = "Coeficiente Aglomerativo (AC)")
    ) |>
    ggplot2::ggplot(ggplot2::aes(x = factor(fold), y = valor, fill = neighborhood)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::geom_hline(yintercept = 0.75, linetype = "dashed", color = "red", linewidth = 0.6) +
    ggplot2::annotate("text", x = 0.6, y = 0.77, label = "0.75", color = "red", size = 3) +
    ggplot2::facet_grid(metrica ~ dataset) +
    ggplot2::scale_fill_brewer(palette = "Set2", name = "Vizinhança") +
    ggplot2::labs(
        title = "Métricas Pré-Corte: CCC e AC por Fold",
        x = "Fold", y = "Valor"
    ) +
    ggplot2::theme_minimal(base_size = 11)

# Gráfico 2: Silhouette e Entropia por (dataset, neighborhood, k, fold)
p_partition <- partition_qual_df |>
    tidyr::pivot_longer(cols = c(silhouette, mean_entropy),
                        names_to = "metrica", values_to = "valor") |>
    dplyr::mutate(
        metrica = dplyr::recode(metrica,
                                silhouette   = "Silhouette medio",
                                mean_entropy = "Entropia media dos labelsets")
    ) |>
    ggplot2::ggplot(ggplot2::aes(
        x     = factor(k),
        y     = valor,
        color = neighborhood,
        group = interaction(neighborhood, fold)
    )) +
    ggplot2::geom_line(alpha = 0.6) +
    ggplot2::geom_point(size = 2) +
    ggplot2::facet_grid(metrica ~ dataset, scales = "free_y") +
    ggplot2::scale_color_brewer(palette = "Set1", name = "Vizinhança") +
    ggplot2::labs(
        title = "Métricas Pós-Corte: Silhouette e Entropia por k e Fold",
        x = "k (número de clusters)", y = "Valor"
    ) +
    ggplot2::theme_minimal(base_size = 11)

# Gráfico 3: Hellinger (k=2) por fold
p_hellinger <- partition_qual_df |>
    dplyr::filter(k == 2) |>
    ggplot2::ggplot(ggplot2::aes(x = factor(fold), y = hellinger, fill = neighborhood)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::facet_wrap(~ dataset) +
    ggplot2::scale_fill_brewer(palette = "Set2", name = "Vizinhança") +
    ggplot2::labs(
        title = "Distância de Hellinger entre Clusters (k=2) por Fold",
        x = "Fold", y = "Hellinger"
    ) +
    ggplot2::theme_minimal(base_size = 11)

# Gráfico 4: Consistência entre folds (mean_sd_props)
p_consistency <- fold_consistency_df |>
    dplyr::select(dataset, neighborhood, k, cluster, mean_sd_props) |>
    dplyr::mutate(painel = sprintf("k=%d | Cluster %d", k, cluster)) |>
    ggplot2::ggplot(ggplot2::aes(x = neighborhood, y = mean_sd_props, fill = neighborhood)) +
    ggplot2::geom_col() +
    ggplot2::facet_grid(dataset ~ painel) +
    ggplot2::scale_fill_brewer(palette = "Set2", name = "Vizinhança") +
    ggplot2::labs(
        title = "Consistência entre Folds: SD médio das proporções de rótulos",
        subtitle = "Valores menores indicam partições mais estáveis",
        x = "Vizinhança", y = "SD médio das proporções"
    ) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))

# Salva PNGs
ggplot2::ggsave(file.path(OUTPUT_PATH, "quality_hierarchy.png"),
                p_hierarchy,   width = 10, height = 6, dpi = 150)
ggplot2::ggsave(file.path(OUTPUT_PATH, "quality_partitions.png"),
                p_partition,   width = 12, height = 7, dpi = 150)
ggplot2::ggsave(file.path(OUTPUT_PATH, "quality_hellinger.png"),
                p_hellinger,   width = 10, height = 5, dpi = 150)
ggplot2::ggsave(file.path(OUTPUT_PATH, "quality_consistency.png"),
                p_consistency, width = 12, height = 6, dpi = 150)

# PDF consolidado de métricas de qualidade
pdf(file.path(OUTPUT_PATH, "quality_summary.pdf"), width = 12, height = 7)
print(p_hierarchy)
print(p_partition)
print(p_hellinger)
print(p_consistency)
dev.off()
```

    png 
      2 

``` r
cat("Graficos de qualidade salvos.\n")
```

    Graficos de qualidade salvos.

## Análise Comparativa da Partição 2

``` r
for (ds in TARGET_DATASETS) {
    label_names <- datasets_config[[ds]]$label_names

    for (nbhd in TARGET_NEIGHBORHOODS) {
        combo  <- sprintf("%s_%s", ds, nbhd)
        p2     <- partition2_data[[combo]]
        if (is.null(p2)) next

        lf  <- p2$label_freq
        lsf <- p2$labelset_freq

        cat(sprintf("\n=== PARTICAO 2: %s | %s ===\n", toupper(ds), toupper(nbhd)))

        cat("\nFrequencia agregada por cluster (soma dos 3 folds):\n")
        lf |>
            dplyr::group_by(cluster) |>
            dplyr::summarise(
                total_instancias = sum(n_instancias),
                dplyr::across(dplyr::all_of(label_names), sum),
                .groups = "drop"
            ) |>
            print()

        p_labels <- lf |>
            tidyr::pivot_longer(
                cols      = dplyr::all_of(label_names),
                names_to  = "rotulo",
                values_to = "frequencia"
            ) |>
            ggplot2::ggplot(ggplot2::aes(
                x    = rotulo,
                y    = frequencia,
                fill = factor(cluster)
            )) +
            ggplot2::geom_col(position = "dodge") +
            ggplot2::facet_wrap(~ paste("Fold", fold)) +
            ggplot2::scale_fill_brewer(palette = "Set1", name = "Cluster") +
            ggplot2::labs(
                title = sprintf("Frequencia de rotulos por cluster — Particao 2\n%s | %s", ds, nbhd),
                x = "Rotulo", y = "Frequencia"
            ) +
            ggplot2::theme_minimal(base_size = 11) +
            ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

        p_labelsets <- lsf |>
            dplyr::group_by(fold, cluster) |>
            dplyr::slice_max(frequencia, n = 8, with_ties = FALSE) |>
            dplyr::ungroup() |>
            dplyr::mutate(painel = sprintf("Fold %d — Cluster %d", fold, cluster)) |>
            ggplot2::ggplot(ggplot2::aes(
                x    = stats::reorder(labelset, frequencia),
                y    = frequencia,
                fill = factor(cluster)
            )) +
            ggplot2::geom_col() +
            ggplot2::coord_flip() +
            ggplot2::facet_wrap(~ painel, scales = "free_y") +
            ggplot2::scale_fill_brewer(palette = "Set1", name = "Cluster") +
            ggplot2::labs(
                title = sprintf("Top 8 labelsets por cluster — Particao 2\n%s | %s", ds, nbhd),
                x = "Labelset (combinacao binaria dos rotulos)", y = "Frequencia"
            ) +
            ggplot2::theme_minimal(base_size = 10) +
            ggplot2::theme(axis.text.y = ggplot2::element_text(size = 8))

        png_labels    <- file.path(OUTPUT_PATH, sprintf("%s_p2_label_freq.png", combo))
        png_labelsets <- file.path(OUTPUT_PATH, sprintf("%s_p2_labelset_freq.png", combo))
        ggplot2::ggsave(png_labels,    p_labels,    width = 12, height = 6,  dpi = 150)
        ggplot2::ggsave(png_labelsets, p_labelsets, width = 14, height = 8,  dpi = 150)

        pdf_analysis <- file.path(OUTPUT_PATH, sprintf("%s_p2_analysis.pdf", combo))
        pdf(pdf_analysis, width = 14, height = 9)
        print(p_labels)
        print(p_labelsets)
        dev.off()

        cat(sprintf("  Outputs da particao 2 salvos para: %s\n", combo))
    }
}
```


    === PARTICAO 2: EMOTIONS | GAUSSIAN ===

    Frequencia agregada por cluster (soma dos 3 folds):
    # A tibble: 2 × 8
      cluster total_instancias amazed.suprised happy.pleased relaxing.calm
        <int>            <int>           <int>         <int>         <int>
    1       1              350              47           103           230
    2       2              243             126            63            34
    # ℹ 3 more variables: quiet.still <int>, sad.lonely <int>,
    #   angry.aggresive <int>

      Outputs da particao 2 salvos para: emotions_gaussian

    === PARTICAO 2: EMOTIONS | BUBBLE ===

    Frequencia agregada por cluster (soma dos 3 folds):
    # A tibble: 2 × 8
      cluster total_instancias amazed.suprised happy.pleased relaxing.calm
        <int>            <int>           <int>         <int>         <int>
    1       1              282              57            55           161
    2       2              311             116           111           103
    # ℹ 3 more variables: quiet.still <int>, sad.lonely <int>,
    #   angry.aggresive <int>

      Outputs da particao 2 salvos para: emotions_bubble

    === PARTICAO 2: SCENE | GAUSSIAN ===

    Frequencia agregada por cluster (soma dos 3 folds):
    # A tibble: 2 × 8
      cluster total_instancias Beach Sunset FallFoliage Field Mountain Urban
        <int>            <int> <int>  <int>       <int> <int>    <int> <int>
    1       1             1698   285    243         384   261      330   286
    2       2              709   142    121          13   172      203   145

      Outputs da particao 2 salvos para: scene_gaussian

    === PARTICAO 2: SCENE | BUBBLE ===

    Frequencia agregada por cluster (soma dos 3 folds):
    # A tibble: 2 × 8
      cluster total_instancias Beach Sunset FallFoliage Field Mountain Urban
        <int>            <int> <int>  <int>       <int> <int>    <int> <int>
    1       1             2043   427      0         397   433      533   431
    2       2              364     0    364           0     0        0     0

      Outputs da particao 2 salvos para: scene_bubble

## Sumário Final dos Outputs

``` r
cat("=== OUTPUTS GERADOS ===\n\n")
```

    === OUTPUTS GERADOS ===

``` r
cat("CSVs — Avaliacao do modelo:\n")
```

    CSVs — Avaliacao do modelo:

``` r
cat("  evaluation_results.csv              — QE por fold, dataset e vizinhanca\n\n")
```

      evaluation_results.csv              — QE por fold, dataset e vizinhanca

``` r
cat("CSVs — Metricas de qualidade do clustering:\n")
```

    CSVs — Metricas de qualidade do clustering:

``` r
cat("  quality_hierarchy.csv               — CCC e AC por fold (pre-corte)\n")
```

      quality_hierarchy.csv               — CCC e AC por fold (pre-corte)

``` r
cat("  quality_partitions.csv              — Silhouette, Entropia e Hellinger por (fold, k)\n")
```

      quality_partitions.csv              — Silhouette, Entropia e Hellinger por (fold, k)

``` r
cat("  quality_fold_consistency.csv        — SD das proporcoes de rotulos entre folds\n\n")
```

      quality_fold_consistency.csv        — SD das proporcoes de rotulos entre folds

``` r
cat("CSVs — Particao 2:\n")
```

    CSVs — Particao 2:

``` r
cat("  {ds}_{nbhd}_p2_label_freq.csv       — frequencia de rotulos na particao 2\n")
```

      {ds}_{nbhd}_p2_label_freq.csv       — frequencia de rotulos na particao 2

``` r
cat("  {ds}_{nbhd}_p2_labelset_freq.csv    — frequencia de labelsets na particao 2\n\n")
```

      {ds}_{nbhd}_p2_labelset_freq.csv    — frequencia de labelsets na particao 2

``` r
cat("PDFs:\n")
```

    PDFs:

``` r
cat("  {ds}_{nbhd}_fold{n}_partitions.pdf  — visualizacoes e metricas de todas as particoes\n")
```

      {ds}_{nbhd}_fold{n}_partitions.pdf  — visualizacoes e metricas de todas as particoes

``` r
cat("  {ds}_{nbhd}_p2_analysis.pdf         — analise comparativa da particao 2\n")
```

      {ds}_{nbhd}_p2_analysis.pdf         — analise comparativa da particao 2

``` r
cat("  quality_summary.pdf                 — sumario visual de todas as metricas de qualidade\n\n")
```

      quality_summary.pdf                 — sumario visual de todas as metricas de qualidade

``` r
cat("PNGs (para o relatorio):\n")
```

    PNGs (para o relatorio):

``` r
cat("  quality_hierarchy.png               — CCC e AC\n")
```

      quality_hierarchy.png               — CCC e AC

``` r
cat("  quality_partitions.png              — Silhouette e Entropia por k\n")
```

      quality_partitions.png              — Silhouette e Entropia por k

``` r
cat("  quality_hellinger.png               — Hellinger (k=2)\n")
```

      quality_hellinger.png               — Hellinger (k=2)

``` r
cat("  quality_consistency.png             — Consistencia entre folds\n")
```

      quality_consistency.png             — Consistencia entre folds

``` r
cat("  {ds}_{nbhd}_p2_label_freq.png       — frequencia de rotulos na particao 2\n")
```

      {ds}_{nbhd}_p2_label_freq.png       — frequencia de rotulos na particao 2

``` r
cat("  {ds}_{nbhd}_p2_labelset_freq.png    — top labelsets na particao 2\n\n")
```

      {ds}_{nbhd}_p2_labelset_freq.png    — top labelsets na particao 2

``` r
cat(sprintf("Diretorio de saida: %s\n", OUTPUT_PATH))
```

    Diretorio de saida: E:/git-tcc/tcc-final/outputs/evaluation

``` r
sessionInfo()
```

    R version 4.4.3 (2025-02-28 ucrt)
    Platform: x86_64-w64-mingw32/x64
    Running under: Windows 11 x64 (build 26200)

    Matrix products: default


    locale:
    [1] LC_COLLATE=Portuguese_Brazil.utf8  LC_CTYPE=Portuguese_Brazil.utf8   
    [3] LC_MONETARY=Portuguese_Brazil.utf8 LC_NUMERIC=C                      
    [5] LC_TIME=Portuguese_Brazil.utf8    

    time zone: America/Sao_Paulo
    tzcode source: internal

    attached base packages:
    [1] grid      stats     graphics  grDevices utils     datasets  methods  
    [8] base     

    other attached packages:
     [1] cluster_2.1.8   gridExtra_2.3   kohonen_3.0.12  lubridate_1.9.4
     [5] forcats_1.0.1   stringr_1.6.0   dplyr_1.1.4     purrr_1.2.1    
     [9] readr_2.1.6     tidyr_1.3.2     tibble_3.3.1    ggplot2_4.0.1  
    [13] tidyverse_2.0.0

    loaded via a namespace (and not attached):
     [1] gtable_0.3.6       jsonlite_2.0.0     compiler_4.4.3     Rcpp_1.1.1        
     [5] tidyselect_1.2.1   textshaping_1.0.4  systemfonts_1.3.1  scales_1.4.0      
     [9] yaml_2.3.12        fastmap_1.2.0      R6_2.6.1           labeling_0.4.3    
    [13] generics_0.1.4     knitr_1.51         pillar_1.11.1      RColorBrewer_1.1-3
    [17] tzdb_0.5.0         rlang_1.1.7        utf8_1.2.6         stringi_1.8.7     
    [21] xfun_0.56          S7_0.2.1           otel_0.2.0         timechange_0.3.0  
    [25] cli_3.6.5          withr_3.0.2        magrittr_2.0.4     digest_0.6.39     
    [29] hms_1.1.4          lifecycle_1.0.5    vctrs_0.7.0        evaluate_1.0.5    
    [33] glue_1.8.0         farver_2.1.2       ragg_1.5.0         rmarkdown_2.30    
    [37] tools_4.4.3        pkgconfig_2.0.3    htmltools_0.5.9   

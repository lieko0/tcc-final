# Script 2 — Avaliação do Modelo e Análise de Partições


## Configuração

``` r
library(tidyverse)
library(kohonen)
library(gridExtra)
library(grid)

GRIDSEARCH_PATH     <- "E:/git-tcc/tcc-final/outputs/gridsearch"
BASE_PATH           <- "E:/git-tcc/tcc-final/data"
OUTPUT_PATH         <- "E:/git-tcc/tcc-final/outputs/evaluation"
SEED                <- 1234L
TARGET_DATASETS     <- c("emotions", "scene")
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
# Na avaliação usamos treino (Tr) e teste (Ts); validação não é mais necessária
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

# best_file <- file.path(GRIDSEARCH_PATH, "best_params_min.csv")
best_file <- file.path(GRIDSEARCH_PATH, "best_params.csv")
if (!file.exists(best_file)) {
    stop(sprintf("Arquivo de melhores parametros nao encontrado: %s\nExecute o script 01_gridsearch.qmd primeiro.", best_file))
}

best_params <- read.csv(
    best_file,
    stringsAsFactors = FALSE
)

cat("Melhores parametros carregados:\n")
```

    Melhores parametros carregados:

``` r
print(best_params)
```

       dataset neighborhood  topology rlen radius alpha_start alpha_end   mean_qe
    1 emotions       bubble hexagonal  500    1.0        0.05      0.01 0.4623441
    2 emotions     gaussian hexagonal 1000    0.5        0.05      0.01 0.5476190
    3    scene       bubble hexagonal 1000    2.0        0.05      0.01 0.3672345
    4    scene     gaussian hexagonal  500    0.5        0.05      0.01 0.4303468
            sd_qe n_folds
    1 0.002592804       3
    2 0.007242258       3
    3 0.011652889       3
    4 0.007797712       3

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

## Treinamento e Avaliação nos 3 Folds

``` r
# Todos os 3 folds de um dado (dataset, vizinhança) usam EXATAMENTE os mesmos parâmetros
# encontrados no grid search — topologia, rlen, radius e alpha incluídos

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


    === EMOTIONS | GAUSSIAN | topology=hexagonal rlen=1000 radius=0.500000 alpha=(0.05, 0.010) ===
      Fold 1: QE_test=0.55199 | neuronios_vazios=0
      Fold 2: QE_test=0.56746 | neuronios_vazios=0
      Fold 3: QE_test=0.53940 | neuronios_vazios=0

    === EMOTIONS | BUBBLE | topology=hexagonal rlen=500 radius=1.000000 alpha=(0.05, 0.010) ===
      Fold 1: QE_test=0.45958 | neuronios_vazios=0
      Fold 2: QE_test=0.46762 | neuronios_vazios=0
      Fold 3: QE_test=0.46240 | neuronios_vazios=0

    === SCENE | GAUSSIAN | topology=hexagonal rlen=500 radius=0.500000 alpha=(0.05, 0.010) ===
      Fold 1: QE_test=0.45182 | neuronios_vazios=0
      Fold 2: QE_test=0.44747 | neuronios_vazios=0
      Fold 3: QE_test=0.43546 | neuronios_vazios=0

    === SCENE | BUBBLE | topology=hexagonal rlen=1000 radius=2.000000 alpha=(0.05, 0.010) ===
      Fold 1: QE_test=0.36884 | neuronios_vazios=0
      Fold 2: QE_test=0.36506 | neuronios_vazios=0
      Fold 3: QE_test=0.36721 | neuronios_vazios=0

``` r
eval_df <- dplyr::bind_rows(eval_results)
write.csv(eval_df, file.path(OUTPUT_PATH, "evaluation_results.csv"), row.names = FALSE)

cat("\n=== RESULTADOS DE AVALIACAO ===\n")
```


    === RESULTADOS DE AVALIACAO ===

``` r
print(eval_df)
```

        dataset neighborhood  topology fold rlen radius alpha_start alpha_end
    1  emotions     gaussian hexagonal    1 1000    0.5        0.05      0.01
    2  emotions     gaussian hexagonal    2 1000    0.5        0.05      0.01
    3  emotions     gaussian hexagonal    3 1000    0.5        0.05      0.01
    4  emotions       bubble hexagonal    1  500    1.0        0.05      0.01
    5  emotions       bubble hexagonal    2  500    1.0        0.05      0.01
    6  emotions       bubble hexagonal    3  500    1.0        0.05      0.01
    7     scene     gaussian hexagonal    1  500    0.5        0.05      0.01
    8     scene     gaussian hexagonal    2  500    0.5        0.05      0.01
    9     scene     gaussian hexagonal    3  500    0.5        0.05      0.01
    10    scene       bubble hexagonal    1 1000    2.0        0.05      0.01
    11    scene       bubble hexagonal    2 1000    2.0        0.05      0.01
    12    scene       bubble hexagonal    3 1000    2.0        0.05      0.01
       empty_neurons   qe_test
    1              0 0.5519924
    2              0 0.5674581
    3              0 0.5393989
    4              0 0.4595837
    5              0 0.4676155
    6              0 0.4623972
    7              0 0.4518221
    8              0 0.4474729
    9              0 0.4354635
    10             0 0.3688435
    11             0 0.3650572
    12             0 0.3672133

## Geração de Partições

``` r
# Para cada (dataset, vizinhança, fold):
#   - Aplica clustering hierárquico sobre os codebook vectors (k = 2 até n_neurons-1)
#   - Para 2x2 = 4 neurônios: k pode ser 2 ou 3
#   - Salva PDF com visualizações de todas as partições
#   - Guarda dados da partição 2 para análise comparativa

partition2_data <- list()

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

            model    <- model_info$model
            Y_tr     <- model_info$Y_tr
            n_neur   <- model$grid$xdim * model$grid$ydim
            max_k    <- n_neur - 1   # Para 2x2: max_k = 3

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

            # Clustering hierárquico sobre os codebook vectors — compartilhado entre ks
            codebook <- model$codes[[1]]
            dend     <- hclust(dist(codebook), method = "complete")

            # --- Partições k = 2 até max_k ---
            for (k in 2:max_k) {
                clusters    <- as.integer(cutree(dend, k = k))
                instance_df <- build_instance_df(model, Y_tr, label_names, clusters)
                label_freq  <- label_freq_by_cluster(instance_df, label_names)
                lset_freq   <- labelset_freq_by_cluster(instance_df, label_names)

                # SOM com cores dos clusters
                plot(model, type = "codes", codeRendering = "segments",
                     bgcol = rainbow(k)[clusters],
                     main  = sprintf("Clusters (k=%d) | %s | %s | Fold %d", k, ds, nbhd, fold))
                kohonen::add.cluster.boundaries(model, clusters)
                pts <- model$grid$pts
                text(pts[, 1], pts[, 2],
                     labels = sprintf("N%d\nC%d", seq_len(nrow(pts)), clusters),
                     cex = 0.75, font = 2)

                # Dendrograma
                plot(dend,
                     main = sprintf("Dendrograma (k=%d) | %s | %s | Fold %d", k, ds, nbhd, fold),
                     xlab = "Neuronios", ylab = "Distancia")
                rect.hclust(dend, k = k, border = "red")

                # Tabela: frequência de rótulos por cluster
                render_table_page(
                    as.data.frame(label_freq),
                    title = sprintf("Frequencia de rotulos por cluster (k=%d) | %s | %s | Fold %d",
                                    k, ds, nbhd, fold)
                )

                # Tabela: top 15 labelsets por cluster
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

                # Guarda dados da partição 2 para análise comparativa posterior
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

        # Consolida dados da partição 2 entre os 3 folds
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

      PDF gerado: emotions_gaussian_fold1_partitions.pdf

      PDF gerado: emotions_gaussian_fold2_partitions.pdf

      PDF gerado: emotions_gaussian_fold3_partitions.pdf
      CSVs da particao 2 salvos para: emotions_gaussian

      PDF gerado: emotions_bubble_fold1_partitions.pdf

      PDF gerado: emotions_bubble_fold2_partitions.pdf

      PDF gerado: emotions_bubble_fold3_partitions.pdf
      CSVs da particao 2 salvos para: emotions_bubble

      PDF gerado: scene_gaussian_fold1_partitions.pdf

      PDF gerado: scene_gaussian_fold2_partitions.pdf

      PDF gerado: scene_gaussian_fold3_partitions.pdf
      CSVs da particao 2 salvos para: scene_gaussian

      PDF gerado: scene_bubble_fold1_partitions.pdf

      PDF gerado: scene_bubble_fold2_partitions.pdf

      PDF gerado: scene_bubble_fold3_partitions.pdf
      CSVs da particao 2 salvos para: scene_bubble

## Análise Comparativa da Partição 2

``` r
# A partição 2 divide os neurônios em 2 grupos.
# Analisamos e comparamos os 3 folds de cada (dataset, vizinhança).

for (ds in TARGET_DATASETS) {
    label_names <- datasets_config[[ds]]$label_names

    for (nbhd in TARGET_NEIGHBORHOODS) {
        combo  <- sprintf("%s_%s", ds, nbhd)
        p2     <- partition2_data[[combo]]
        if (is.null(p2)) next

        lf  <- p2$label_freq
        lsf <- p2$labelset_freq

        cat(sprintf("\n=== PARTICAO 2: %s | %s ===\n", toupper(ds), toupper(nbhd)))

        # Sumário: total de instâncias e rótulos por cluster (agregado entre folds)
        cat("\nFrequencia agregada por cluster (soma dos 3 folds):\n")
        lf |>
            dplyr::group_by(cluster) |>
            dplyr::summarise(
                total_instancias = sum(n_instancias),
                dplyr::across(dplyr::all_of(label_names), sum),
                .groups = "drop"
            ) |>
            print()

        # --- Gráfico 1: frequência de rótulos por cluster e fold ---
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

        # --- Gráfico 2: top labelsets por cluster e fold ---
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

        # Salva PNGs para o relatório
        png_labels    <- file.path(OUTPUT_PATH, sprintf("%s_p2_label_freq.png", combo))
        png_labelsets <- file.path(OUTPUT_PATH, sprintf("%s_p2_labelset_freq.png", combo))
        ggplot2::ggsave(png_labels,    p_labels,    width = 12, height = 6,  dpi = 150)
        ggplot2::ggsave(png_labelsets, p_labelsets, width = 14, height = 8,  dpi = 150)

        # PDF de análise comparativa da partição 2
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
    1       1              316             114            73           117
    2       2              277              59            93           147
    # ℹ 3 more variables: quiet.still <int>, sad.lonely <int>,
    #   angry.aggresive <int>

      Outputs da particao 2 salvos para: emotions_gaussian

    === PARTICAO 2: EMOTIONS | BUBBLE ===

    Frequencia agregada por cluster (soma dos 3 folds):
    # A tibble: 2 × 8
      cluster total_instancias amazed.suprised happy.pleased relaxing.calm
        <int>            <int>           <int>         <int>         <int>
    1       1              307              57            92           169
    2       2              286             116            74            95
    # ℹ 3 more variables: quiet.still <int>, sad.lonely <int>,
    #   angry.aggresive <int>

      Outputs da particao 2 salvos para: emotions_bubble

    === PARTICAO 2: SCENE | GAUSSIAN ===

    Frequencia agregada por cluster (soma dos 3 folds):
    # A tibble: 2 × 8
      cluster total_instancias Beach Sunset FallFoliage Field Mountain Urban
        <int>            <int> <int>  <int>       <int> <int>    <int> <int>
    1       1             1964   408    242         392   405      355   288
    2       2              443    19    122           5    28      178   143

      Outputs da particao 2 salvos para: scene_gaussian

    === PARTICAO 2: SCENE | BUBBLE ===

    Frequencia agregada por cluster (soma dos 3 folds):
    # A tibble: 2 × 8
      cluster total_instancias Beach Sunset FallFoliage Field Mountain Urban
        <int>            <int> <int>  <int>       <int> <int>    <int> <int>
    1       1             1460   278    243         264   286      354   143
    2       2              947   149    121         133   147      179   288

      Outputs da particao 2 salvos para: scene_bubble

## Sumário Final dos Outputs

``` r
cat("=== OUTPUTS GERADOS ===\n\n")
```

    === OUTPUTS GERADOS ===

``` r
cat("CSVs:\n")
```

    CSVs:

``` r
cat("  evaluation_results.csv            — QE por fold, dataset e vizinhanca\n")
```

      evaluation_results.csv            — QE por fold, dataset e vizinhanca

``` r
cat("  {ds}_{nbhd}_p2_label_freq.csv     — frequencia de rotulos na particao 2\n")
```

      {ds}_{nbhd}_p2_label_freq.csv     — frequencia de rotulos na particao 2

``` r
cat("  {ds}_{nbhd}_p2_labelset_freq.csv  — frequencia de labelsets na particao 2\n\n")
```

      {ds}_{nbhd}_p2_labelset_freq.csv  — frequencia de labelsets na particao 2

``` r
cat("PDFs:\n")
```

    PDFs:

``` r
cat("  {ds}_{nbhd}_fold{n}_partitions.pdf — visualizacoes de todas as particoes por fold\n")
```

      {ds}_{nbhd}_fold{n}_partitions.pdf — visualizacoes de todas as particoes por fold

``` r
cat("  {ds}_{nbhd}_p2_analysis.pdf        — analise comparativa da particao 2\n\n")
```

      {ds}_{nbhd}_p2_analysis.pdf        — analise comparativa da particao 2

``` r
cat("PNGs (para o relatorio):\n")
```

    PNGs (para o relatorio):

``` r
cat("  {ds}_{nbhd}_p2_label_freq.png     — grafico de frequencia de rotulos\n")
```

      {ds}_{nbhd}_p2_label_freq.png     — grafico de frequencia de rotulos

``` r
cat("  {ds}_{nbhd}_p2_labelset_freq.png  — grafico de top labelsets\n\n")
```

      {ds}_{nbhd}_p2_labelset_freq.png  — grafico de top labelsets

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
     [1] gridExtra_2.3   kohonen_3.0.12  lubridate_1.9.4 forcats_1.0.1  
     [5] stringr_1.6.0   dplyr_1.1.4     purrr_1.2.1     readr_2.1.6    
     [9] tidyr_1.3.2     tibble_3.3.1    ggplot2_4.0.1   tidyverse_2.0.0

    loaded via a namespace (and not attached):
     [1] gtable_0.3.6       jsonlite_2.0.0     compiler_4.4.3     Rcpp_1.1.1        
     [5] tidyselect_1.2.1   textshaping_1.0.4  systemfonts_1.3.1  scales_1.4.0      
     [9] yaml_2.3.12        fastmap_1.2.0      R6_2.6.1           labeling_0.4.3    
    [13] generics_0.1.4     knitr_1.51         pillar_1.11.1      RColorBrewer_1.1-3
    [17] tzdb_0.5.0         rlang_1.1.7        stringi_1.8.7      xfun_0.56         
    [21] S7_0.2.1           otel_0.2.0         timechange_0.3.0   cli_3.6.5         
    [25] withr_3.0.2        magrittr_2.0.4     digest_0.6.39      hms_1.1.4         
    [29] lifecycle_1.0.5    vctrs_0.7.0        evaluate_1.0.5     glue_1.8.0        
    [33] farver_2.1.2       ragg_1.5.0         rmarkdown_2.30     tools_4.4.3       
    [37] pkgconfig_2.0.3    htmltools_0.5.9   

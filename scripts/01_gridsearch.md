# Script 1 — Grid Search


## Configuração

``` r
library(tidyverse)
library(kohonen)

BASE_PATH <- "E:/git-tcc/tcc-final/data"
OUTPUT_PATH <- "E:/git-tcc/tcc-final/outputs/gridsearch"
SEED <- 1234L # a letra "L" indica que se trata de um número inteiro
TARGET_DATASETS <- c("emotions", "scene")

dir.create(OUTPUT_PATH, recursive = TRUE, showWarnings = FALSE)
set.seed(SEED)
```

## Configuração dos Datasets

``` r
datasets_config <- list(
    emotions = list(
        attr_cols = 1:72,
        label_cols = 73:78,
        label_names = c(
            "amazed.suprised", "happy.pleased", "relaxing.calm",
            "quiet.still", "sad.lonely", "angry.aggresive"
        )
    ),
    scene = list(
        attr_cols   = 1:294,
        label_cols  = 295:300,
        label_names = c("Beach", "Sunset", "FallFoliage", "Field", "Mountain", "Urban")
    )
)
```

## Carregamento dos Dados

``` r
# O gridsearch usa apenas treino (Tr) e validação (Vl) — nunca o teste
load_fold <- function(base_path, dataset_name, config, fold, types = c("Tr", "Vl")) {
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

cat("Datasets carregados:\n")
```

    Datasets carregados:

``` r
for (ds in TARGET_DATASETS) {
    cfg <- datasets_config[[ds]]
    cat(sprintf(
        "  %-10s — %d atributos, %d rotulos\n",
        ds, length(cfg$attr_cols), length(cfg$label_cols)
    ))
}
```

      emotions   — 72 atributos, 6 rotulos
      scene      — 294 atributos, 6 rotulos

``` r
dplyr::glimpse(datasets$emotions$fold1$tr$labels)
```

    Rows: 197
    Columns: 6
    $ amazed.suprised <int> 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, …
    $ happy.pleased   <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, …
    $ relaxing.calm   <int> 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, …
    $ quiet.still     <int> 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, …
    $ sad.lonely      <int> 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, …
    $ angry.aggresive <int> 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, …

``` r
dplyr::glimpse(datasets$scene$fold1$tr$labels)
```

    Rows: 802
    Columns: 6
    $ Beach       <int> 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0…
    $ Sunset      <int> 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1…
    $ FallFoliage <int> 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0…
    $ Field       <int> 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0…
    $ Mountain    <int> 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0…
    $ Urban       <int> 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0…

## Definição das Grades SOM (2×2 apenas)

``` r
# apenas grade 2x2, 2 vizinhancas x 2 topologias = 4 grids
grids <- list(
    gaussian_rectangular = kohonen::somgrid(2, 2, "rectangular", neighbourhood.fct = "gaussian"),
    gaussian_hexagonal   = kohonen::somgrid(2, 2, "hexagonal", neighbourhood.fct = "gaussian"),
    bubble_rectangular   = kohonen::somgrid(2, 2, "rectangular", neighbourhood.fct = "bubble"),
    bubble_hexagonal     = kohonen::somgrid(2, 2, "hexagonal", neighbourhood.fct = "bubble")
)

cat("Grades definidas:\n")
```

    Grades definidas:

``` r
for (name in names(grids)) {
    g <- grids[[name]]
    cat(sprintf(
        "  %-28s — %dx%d | topo=%-15s | viz=%s\n",
        name, g$xdim, g$ydim, g$topo, g$neighbourhood.fct
    ))
}
```

      gaussian_rectangular         — 2x2 | topo=rectangular     | viz=gaussian
      gaussian_hexagonal           — 2x2 | topo=hexagonal       | viz=gaussian
      bubble_rectangular           — 2x2 | topo=rectangular     | viz=bubble
      bubble_hexagonal             — 2x2 | topo=hexagonal       | viz=bubble

## Grade de Parâmetros

``` r
param_grid <- expand.grid(
    rlen = c(100L, 500L, 1000L), # default: 100L (quanto mais iterações, melhor convergência, mas mais tempo)
    radius = c(1.0, 1.5), # default: 1.0 (como o grid é 2x2, não faz sentido usar um raio maior que isso pois diagonal é ~1.41)
    alpha_start = c(0.05, 0.1), # default: 0.05
    alpha_end = c(0.005, 0.01), # default: 0.01
    grid_name = names(grids),
    dataset = TARGET_DATASETS,
    fold = 1:3,
    stringsAsFactors = FALSE
) |>
    dplyr::filter(alpha_start > alpha_end)

cat(sprintf(
    "Total de combinacoes: %d\n  (%d grids x %d datasets x %d folds x %d combinacoes de params)\n",
    nrow(param_grid),
    length(grids),
    length(TARGET_DATASETS),
    3,
    length(unique(param_grid$rlen)) * length(unique(param_grid$radius)) *
        length(unique(param_grid$alpha_start)) * length(unique(param_grid$alpha_end))
))
```

    Total de combinacoes: 576
      (4 grids x 2 datasets x 3 folds x 24 combinacoes de params)

## Execução do Grid Search

``` r
results_list <- vector("list", nrow(param_grid))
start_time <- proc.time()

for (i in seq_len(nrow(param_grid))) {
    p <- param_grid[i, ]
    grid <- grids[[p$grid_name]]

    Y_tr <- as.matrix(datasets[[p$dataset]][[paste0("fold", p$fold)]][["tr"]]$labels)
    Y_vl <- as.matrix(datasets[[p$dataset]][[paste0("fold", p$fold)]][["vl"]]$labels)

    model <- tryCatch(
        {
            kohonen::supersom(
                data      = list(Y = Y_tr),
                grid      = grid,
                rlen      = p$rlen,
                radius    = p$radius,
                alpha     = c(p$alpha_start, p$alpha_end),
                keep.data = TRUE
            )
        },
        error = function(e) {
            message("Erro (i=", i, "): ", e$message)
            NULL
        }
    )

    if (is.null(model)) {
        qe_vl <- NA_real_
        empty_n <- NA_integer_
    } else {
        grid_size <- grid$xdim * grid$ydim
        empty_n <- grid_size - length(unique(model$unit.classif))
        qe_vl <- mean(kohonen::map(model, newdata = list(Y = Y_vl))$distances)
    }

    results_list[[i]] <- data.frame(
        dataset       = p$dataset,
        fold          = p$fold,
        neighborhood  = grid$neighbourhood.fct,
        topology      = grid$topo,
        grid_size     = sprintf("%dx%d", grid$xdim, grid$ydim),
        rlen          = p$rlen,
        radius        = p$radius,
        alpha_start   = p$alpha_start,
        alpha_end     = p$alpha_end,
        empty_neurons = empty_n,
        qe_val        = qe_vl
    )

    elapsed <- round((proc.time() - start_time)[[3]], 1)
    cat(sprintf(
        "[%7.1fs] %4d/%d | %-9s fold%d | %-24s | rlen=%4d r=%f a=(%.2f,%.3f) | empty=%s QE=%s\n",
        elapsed, i, nrow(param_grid),
        p$dataset, p$fold, p$grid_name,
        p$rlen, p$radius, p$alpha_start, p$alpha_end,
        ifelse(is.na(empty_n), "NA", empty_n),
        ifelse(is.na(qe_vl), "NA", sprintf("%.5f", qe_vl))
    ))
}
```

    [    0.2s]    1/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.58947
    [    0.2s]    2/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.60297
    [    0.4s]    3/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.58452
    [    0.4s]    4/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.57278
    [    0.4s]    5/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.59292
    [    0.6s]    6/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.60109
    [    0.6s]    7/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.58405
    [    0.7s]    8/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.58106
    [    0.7s]    9/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57628
    [    0.8s]   10/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.60340
    [    0.8s]   11/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.58918
    [    0.9s]   12/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.57492
    [    0.9s]   13/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58126
    [    0.9s]   14/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.60385
    [    1.0s]   15/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57923
    [    1.0s]   16/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.60119
    [    1.1s]   17/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.58631
    [    1.2s]   18/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59192
    [    1.3s]   19/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.58184
    [    1.3s]   20/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.58565
    [    1.4s]   21/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.58055
    [    1.5s]   22/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.58727
    [    1.5s]   23/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.58196
    [    1.6s]   24/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.57241
    [    1.6s]   25/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.59885
    [    1.7s]   26/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.58747
    [    1.8s]   27/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.56442
    [    1.8s]   28/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.57297
    [    1.9s]   29/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.55395
    [    1.9s]   30/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.54516
    [    2.0s]   31/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.55897
    [    2.0s]   32/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.54414
    [    2.1s]   33/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.54744
    [    2.1s]   34/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.54472
    [    2.2s]   35/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.55074
    [    2.3s]   36/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.55720
    [    2.3s]   37/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56643
    [    2.4s]   38/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55016
    [    2.5s]   39/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.54718
    [    2.5s]   40/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.56261
    [    2.5s]   41/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.56228
    [    2.6s]   42/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54419
    [    2.6s]   43/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.55823
    [    2.6s]   44/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.55115
    [    2.7s]   45/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.54507
    [    2.7s]   46/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54615
    [    2.8s]   47/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.55914
    [    2.8s]   48/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54736
    [    2.8s]   49/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.49540
    [    2.8s]   50/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.50326
    [    2.9s]   51/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46588
    [    2.9s]   52/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.50817
    [    2.9s]   53/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.48502
    [    2.9s]   54/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45848
    [    3.0s]   55/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46714
    [    3.0s]   56/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.48117
    [    3.0s]   57/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45838
    [    3.0s]   58/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.48262
    [    3.1s]   59/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45927
    [    3.1s]   60/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45895
    [    3.1s]   61/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.50928
    [    3.1s]   62/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49606
    [    3.2s]   63/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46671
    [    3.2s]   64/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48128
    [    3.2s]   65/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46628
    [    3.2s]   66/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45904
    [    3.2s]   67/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45974
    [    3.2s]   68/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45997
    [    3.3s]   69/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46405
    [    3.3s]   70/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.49608
    [    3.3s]   71/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46150
    [    3.3s]   72/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45905
    [    3.4s]   73/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.49530
    [    3.4s]   74/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.48157
    [    3.4s]   75/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.50848
    [    3.5s]   76/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.48383
    [    3.5s]   77/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.50223
    [    3.5s]   78/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45852
    [    3.6s]   79/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45935
    [    3.6s]   80/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46619
    [    3.6s]   81/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45839
    [    3.6s]   82/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46685
    [    3.7s]   83/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46006
    [    3.8s]   84/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46048
    [    3.8s]   85/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46007
    [    3.8s]   86/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.50498
    [    3.9s]   87/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46586
    [    3.9s]   88/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.52202
    [    3.9s]   89/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49640
    [    3.9s]   90/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48562
    [    3.9s]   91/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46887
    [    4.0s]   92/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.48211
    [    4.0s]   93/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.48336
    [    4.0s]   94/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45960
    [    4.1s]   95/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.48462
    [    4.1s]   96/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46684
    [    4.2s]   97/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44836
    [    4.4s]   98/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44930
    [    4.7s]   99/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45709
    [    4.8s]  100/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44819
    [    4.9s]  101/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44831
    [    5.3s]  102/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45070
    [    5.3s]  103/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45277
    [    5.5s]  104/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43875
    [    5.8s]  105/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45160
    [    5.8s]  106/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45195
    [    6.0s]  107/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.44767
    [    6.3s]  108/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.44869
    [    6.3s]  109/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44732
    [    6.4s]  110/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46105
    [    6.8s]  111/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44744
    [    6.8s]  112/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44964
    [    7.0s]  113/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45004
    [    7.3s]  114/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46817
    [    7.3s]  115/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44672
    [    7.4s]  116/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45492
    [    7.7s]  117/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45218
    [    7.8s]  118/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45225
    [    8.0s]  119/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.44157
    [    8.3s]  120/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45487
    [    8.3s]  121/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44031
    [    8.4s]  122/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.43036
    [    8.7s]  123/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.43340
    [    8.7s]  124/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43715
    [    8.9s]  125/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.42973
    [    9.2s]  126/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45373
    [    9.2s]  127/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43472
    [    9.4s]  128/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43822
    [    9.6s]  129/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.42607
    [    9.6s]  130/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.44552
    [    9.8s]  131/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.43289
    [   10.0s]  132/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.42751
    [   10.1s]  133/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46975
    [   10.2s]  134/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43538
    [   10.5s]  135/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43567
    [   10.5s]  136/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43598
    [   10.6s]  137/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44366
    [   10.9s]  138/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43613
    [   10.9s]  139/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44314
    [   11.1s]  140/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44137
    [   11.3s]  141/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.43893
    [   11.4s]  142/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.43481
    [   11.5s]  143/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.42931
    [   11.8s]  144/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.42878
    [   11.8s]  145/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37576
    [   11.9s]  146/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37856
    [   12.0s]  147/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.50918
    [   12.1s]  148/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37207
    [   12.1s]  149/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36682
    [   12.3s]  150/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36634
    [   12.4s]  151/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.37429
    [   12.4s]  152/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.36056
    [   12.6s]  153/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.37288
    [   12.6s]  154/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.36554
    [   12.7s]  155/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.36212
    [   12.9s]  156/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.35801
    [   12.9s]  157/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36986
    [   13.0s]  158/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37825
    [   13.1s]  159/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38063
    [   13.1s]  160/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36400
    [   13.3s]  161/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36333
    [   13.5s]  162/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37225
    [   13.5s]  163/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38164
    [   13.6s]  164/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38065
    [   13.7s]  165/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.52771
    [   13.8s]  166/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.35828
    [   13.9s]  167/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36716
    [   14.1s]  168/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.38400
    [   14.1s]  169/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.36890
    [   14.3s]  170/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.36918
    [   14.4s]  171/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.39010
    [   14.4s]  172/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37345
    [   14.5s]  173/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36211
    [   14.6s]  174/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37152
    [   14.7s]  175/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.36007
    [   14.8s]  176/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.36906
    [   14.9s]  177/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.38073
    [   14.9s]  178/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.37845
    [   15.0s]  179/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.35816
    [   15.2s]  180/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.38037
    [   15.2s]  181/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37157
    [   15.3s]  182/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=1 QE=0.55475
    [   15.5s]  183/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37179
    [   15.6s]  184/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38329
    [   15.7s]  185/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.35845
    [   15.9s]  186/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37937
    [   15.9s]  187/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38162
    [   16.0s]  188/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38080
    [   16.1s]  189/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37924
    [   16.2s]  190/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36649
    [   16.3s]  191/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37222
    [   16.5s]  192/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37340
    [   16.5s]  193/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.58783
    [   16.5s]  194/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.58717
    [   16.6s]  195/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.57961
    [   16.6s]  196/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.62206
    [   16.6s]  197/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.58970
    [   16.7s]  198/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.58897
    [   16.7s]  199/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57741
    [   16.7s]  200/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57840
    [   16.8s]  201/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57468
    [   16.8s]  202/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.58150
    [   16.9s]  203/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.58605
    [   17.0s]  204/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.57168
    [   17.0s]  205/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57016
    [   17.1s]  206/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57359
    [   17.2s]  207/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57853
    [   17.2s]  208/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.57403
    [   17.2s]  209/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.58565
    [   17.3s]  210/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.57672
    [   17.3s]  211/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.58284
    [   17.3s]  212/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.57120
    [   17.4s]  213/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.57443
    [   17.4s]  214/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.58993
    [   17.4s]  215/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.57546
    [   17.5s]  216/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.57805
    [   17.5s]  217/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.57565
    [   17.6s]  218/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.56038
    [   17.6s]  219/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.56207
    [   17.6s]  220/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.53614
    [   17.7s]  221/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.54708
    [   17.7s]  222/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.54146
    [   17.8s]  223/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.55961
    [   17.9s]  224/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.53923
    [   18.0s]  225/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.53975
    [   18.0s]  226/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.54333
    [   18.0s]  227/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.53286
    [   18.1s]  228/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.53427
    [   18.1s]  229/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56479
    [   18.1s]  230/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57546
    [   18.2s]  231/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.54596
    [   18.2s]  232/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54433
    [   18.3s]  233/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54963
    [   18.3s]  234/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54298
    [   18.4s]  235/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.55176
    [   18.4s]  236/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.56187
    [   18.4s]  237/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.55869
    [   18.5s]  238/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54079
    [   18.5s]  239/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.55164
    [   18.6s]  240/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54293
    [   18.6s]  241/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46272
    [   18.6s]  242/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46212
    [   18.6s]  243/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.47629
    [   18.6s]  244/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.46152
    [   18.6s]  245/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.47955
    [   18.7s]  246/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.48169
    [   18.7s]  247/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.47626
    [   18.7s]  248/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46145
    [   18.7s]  249/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.47541
    [   18.7s]  250/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.49445
    [   18.8s]  251/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46161
    [   18.8s]  252/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46100
    [   18.8s]  253/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.47767
    [   18.8s]  254/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46417
    [   18.9s]  255/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46186
    [   18.9s]  256/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.47999
    [   18.9s]  257/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46101
    [   19.0s]  258/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48118
    [   19.0s]  259/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46407
    [   19.0s]  260/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46152
    [   19.0s]  261/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.48137
    [   19.0s]  262/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46241
    [   19.0s]  263/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45889
    [   19.0s]  264/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46334
    [   19.1s]  265/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46173
    [   19.1s]  266/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45783
    [   19.1s]  267/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46065
    [   19.1s]  268/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.49261
    [   19.1s]  269/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.47930
    [   19.2s]  270/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45636
    [   19.2s]  271/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.47884
    [   19.2s]  272/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.47697
    [   19.2s]  273/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45838
    [   19.2s]  274/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.49371
    [   19.2s]  275/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.47951
    [   19.3s]  276/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.49274
    [   19.3s]  277/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46209
    [   19.3s]  278/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46237
    [   19.3s]  279/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45729
    [   19.4s]  280/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46061
    [   19.4s]  281/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45816
    [   19.4s]  282/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48127
    [   19.4s]  283/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46162
    [   19.4s]  284/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46470
    [   19.5s]  285/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46003
    [   19.5s]  286/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.49498
    [   19.5s]  287/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.49267
    [   19.5s]  288/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45930
    [   19.6s]  289/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44901
    [   19.7s]  290/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45708
    [   20.0s]  291/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45426
    [   20.1s]  292/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44182
    [   20.3s]  293/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44797
    [   20.6s]  294/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.47004
    [   20.6s]  295/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45850
    [   20.8s]  296/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45733
    [   21.0s]  297/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46159
    [   21.0s]  298/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45439
    [   21.2s]  299/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45444
    [   21.4s]  300/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45169
    [   21.5s]  301/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44691
    [   21.6s]  302/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.47204
    [   21.8s]  303/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45264
    [   21.9s]  304/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45962
    [   22.0s]  305/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46174
    [   22.3s]  306/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45483
    [   22.3s]  307/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46385
    [   22.4s]  308/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45431
    [   22.7s]  309/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44117
    [   22.8s]  310/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45062
    [   22.9s]  311/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.44256
    [   23.1s]  312/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.44022
    [   23.2s]  313/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.43844
    [   23.3s]  314/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45166
    [   23.6s]  315/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44162
    [   23.6s]  316/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44574
    [   23.8s]  317/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43046
    [   24.0s]  318/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43983
    [   24.1s]  319/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43707
    [   24.2s]  320/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.44407
    [   24.5s]  321/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43412
    [   24.5s]  322/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.42502
    [   24.7s]  323/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.43545
    [   25.0s]  324/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.43412
    [   25.0s]  325/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46532
    [   25.2s]  326/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43957
    [   25.4s]  327/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.42339
    [   25.4s]  328/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44580
    [   25.6s]  329/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43825
    [   25.8s]  330/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43318
    [   25.9s]  331/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.42505
    [   26.0s]  332/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.43066
    [   26.2s]  333/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44182
    [   26.3s]  334/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45688
    [   26.4s]  335/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.42820
    [   26.7s]  336/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.43478
    [   26.8s]  337/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.38200
    [   26.8s]  338/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.39360
    [   27.0s]  339/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37173
    [   27.0s]  340/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37052
    [   27.1s]  341/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36744
    [   27.3s]  342/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.35893
    [   27.3s]  343/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.52640
    [   27.4s]  344/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.38324
    [   27.6s]  345/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.36996
    [   27.6s]  346/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.38796
    [   27.7s]  347/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.35841
    [   27.9s]  348/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.39125
    [   27.9s]  349/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.53844
    [   28.0s]  350/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38912
    [   28.2s]  351/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38772
    [   28.2s]  352/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.39514
    [   28.3s]  353/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36616
    [   28.5s]  354/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37358
    [   28.5s]  355/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38203
    [   28.6s]  356/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.36087
    [   28.8s]  357/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37407
    [   28.8s]  358/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36036
    [   28.9s]  359/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36317
    [   29.1s]  360/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36093
    [   29.1s]  361/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.36110
    [   29.2s]  362/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.38253
    [   29.3s]  363/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37072
    [   29.4s]  364/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.38373
    [   29.4s]  365/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37044
    [   29.6s]  366/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.38390
    [   29.6s]  367/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=1 QE=0.53100
    [   29.8s]  368/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.38152
    [   29.9s]  369/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.37187
    [   29.9s]  370/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.38234
    [   30.0s]  371/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.36611
    [   30.2s]  372/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.38102
    [   30.2s]  373/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36133
    [   30.3s]  374/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.39194
    [   30.5s]  375/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.39147
    [   30.5s]  376/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38359
    [   30.6s]  377/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38200
    [   30.8s]  378/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38399
    [   30.9s]  379/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37402
    [   31.0s]  380/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37209
    [   31.1s]  381/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.36244
    [   31.1s]  382/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37951
    [   31.2s]  383/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.38070
    [   31.4s]  384/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37020
    [   31.4s]  385/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.57692
    [   31.4s]  386/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.57682
    [   31.5s]  387/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.57684
    [   31.5s]  388/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.59517
    [   31.6s]  389/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.58286
    [   31.6s]  390/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.59943
    [   31.7s]  391/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57228
    [   31.7s]  392/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57371
    [   31.8s]  393/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57371
    [   31.8s]  394/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.58683
    [   31.8s]  395/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.61164
    [   31.9s]  396/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.57337
    [   31.9s]  397/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58162
    [   31.9s]  398/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.60969
    [   32.0s]  399/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57353
    [   32.0s]  400/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.60252
    [   32.1s]  401/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.60017
    [   32.1s]  402/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.57500
    [   32.1s]  403/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.56516
    [   32.2s]  404/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.57374
    [   32.3s]  405/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.57305
    [   32.3s]  406/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.60688
    [   32.3s]  407/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.59848
    [   32.4s]  408/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.58503
    [   32.4s]  409/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.54509
    [   32.5s]  410/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.54065
    [   32.5s]  411/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.54448
    [   32.6s]  412/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.54074
    [   32.6s]  413/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.55618
    [   32.7s]  414/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.55621
    [   32.7s]  415/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.54214
    [   32.8s]  416/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.54400
    [   32.8s]  417/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.53601
    [   32.8s]  418/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.55079
    [   32.9s]  419/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.54335
    [   33.0s]  420/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.53385
    [   33.0s]  421/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58595
    [   33.0s]  422/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56973
    [   33.1s]  423/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55346
    [   33.1s]  424/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.55379
    [   33.1s]  425/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54043
    [   33.2s]  426/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.55569
    [   33.3s]  427/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.54339
    [   33.3s]  428/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.54222
    [   33.4s]  429/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.54582
    [   33.4s]  430/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54012
    [   33.5s]  431/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54227
    [   33.5s]  432/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.53438
    [   33.5s]  433/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46484
    [   33.6s]  434/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46004
    [   33.6s]  435/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.49798
    [   33.6s]  436/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.48018
    [   33.6s]  437/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.46503
    [   33.7s]  438/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45904
    [   33.7s]  439/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46196
    [   33.7s]  440/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45907
    [   33.8s]  441/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46710
    [   33.8s]  442/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.48435
    [   33.8s]  443/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.48304
    [   33.8s]  444/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45817
    [   33.8s]  445/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45849
    [   33.8s]  446/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49895
    [   33.9s]  447/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46038
    [   33.9s]  448/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48237
    [   33.9s]  449/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48482
    [   34.0s]  450/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46012
    [   34.0s]  451/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.48517
    [   34.0s]  452/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46190
    [   34.1s]  453/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46033
    [   34.1s]  454/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45857
    [   34.1s]  455/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.48100
    [   34.1s]  456/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.48534
    [   34.1s]  457/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.48216
    [   34.2s]  458/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45845
    [   34.2s]  459/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45932
    [   34.2s]  460/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.46066
    [   34.2s]  461/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.49750
    [   34.3s]  462/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.49901
    [   34.3s]  463/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.49918
    [   34.3s]  464/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45983
    [   34.4s]  465/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46639
    [   34.4s]  466/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.49713
    [   34.4s]  467/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46532
    [   34.4s]  468/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.49795
    [   34.4s]  469/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48124
    [   34.5s]  470/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48211
    [   34.5s]  471/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46110
    [   34.5s]  472/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46012
    [   34.5s]  473/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49701
    [   34.6s]  474/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.52483
    [   34.6s]  475/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.48083
    [   34.6s]  476/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46260
    [   34.6s]  477/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46856
    [   34.6s]  478/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.49802
    [   34.6s]  479/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.49879
    [   34.7s]  480/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46199
    [   34.7s]  481/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45639
    [   34.9s]  482/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45193
    [   35.1s]  483/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45391
    [   35.1s]  484/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44992
    [   35.3s]  485/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45710
    [   35.6s]  486/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45357
    [   35.6s]  487/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.44768
    [   35.7s]  488/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.44959
    [   36.0s]  489/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45727
    [   36.0s]  490/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45201
    [   36.1s]  491/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45216
    [   36.5s]  492/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.44040
    [   36.5s]  493/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.47718
    [   36.6s]  494/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45860
    [   36.9s]  495/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45649
    [   36.9s]  496/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46383
    [   37.1s]  497/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45315
    [   37.4s]  498/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45257
    [   37.4s]  499/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44740
    [   37.6s]  500/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45786
    [   37.8s]  501/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44049
    [   37.9s]  502/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45616
    [   38.1s]  503/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46670
    [   38.4s]  504/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.44359
    [   38.4s]  505/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.42722
    [   38.6s]  506/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44406
    [   38.9s]  507/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.43043
    [   38.9s]  508/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43985
    [   39.1s]  509/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43463
    [   39.3s]  510/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43389
    [   39.4s]  511/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.42844
    [   39.5s]  512/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.44337
    [   39.8s]  513/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43196
    [   39.8s]  514/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.42947
    [   40.0s]  515/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.42681
    [   40.2s]  516/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.44031
    [   40.3s]  517/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44275
    [   40.4s]  518/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45918
    [   40.6s]  519/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44045
    [   40.7s]  520/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45676
    [   40.8s]  521/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43897
    [   41.1s]  522/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43213
    [   41.1s]  523/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.43341
    [   41.3s]  524/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44555
    [   41.5s]  525/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44417
    [   41.5s]  526/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.43478
    [   41.6s]  527/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.42592
    [   41.9s]  528/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.42247
    [   41.9s]  529/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37898
    [   42.0s]  530/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.35849
    [   42.1s]  531/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.38192
    [   42.2s]  532/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37458
    [   42.3s]  533/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.35780
    [   42.5s]  534/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36762
    [   42.5s]  535/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.35846
    [   42.5s]  536/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=1 QE=0.52915
    [   42.6s]  537/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.38084
    [   42.7s]  538/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.37259
    [   42.8s]  539/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.37216
    [   43.0s]  540/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.35842
    [   43.0s]  541/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36938
    [   43.1s]  542/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.39398
    [   43.2s]  543/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37276
    [   43.3s]  544/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36413
    [   43.4s]  545/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36332
    [   43.5s]  546/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38365
    [   43.6s]  547/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37338
    [   43.6s]  548/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38068
    [   43.8s]  549/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.50981
    [   43.8s]  550/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37852
    [   43.9s]  551/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36786
    [   44.1s]  552/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37016
    [   44.1s]  553/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37101
    [   44.2s]  554/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37042
    [   44.3s]  555/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37141
    [   44.3s]  556/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36362
    [   44.4s]  557/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37316
    [   44.5s]  558/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37020
    [   44.6s]  559/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.37162
    [   44.7s]  560/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.38410
    [   44.8s]  561/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.37369
    [   44.8s]  562/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.36318
    [   45.0s]  563/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.36868
    [   45.1s]  564/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.38222
    [   45.1s]  565/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38185
    [   45.2s]  566/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37301
    [   45.3s]  567/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36130
    [   45.3s]  568/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38050
    [   45.4s]  569/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38350
    [   45.6s]  570/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36617
    [   45.6s]  571/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38273
    [   45.7s]  572/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38540
    [   45.8s]  573/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37364
    [   45.9s]  574/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36279
    [   46.0s]  575/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37096
    [   46.1s]  576/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.39440

``` r
results_df <- dplyr::bind_rows(results_list)
results_csv <- file.path(OUTPUT_PATH, sprintf("gridsearch_seed%d.csv", SEED))
write.csv(results_df, results_csv, row.names = FALSE)
cat(sprintf("\nResultados brutos salvos em: %s\n", results_csv))
```


    Resultados brutos salvos em: E:/git-tcc/tcc-final/outputs/gridsearch/gridsearch_seed1234.csv

## Sumário do Grid Search

``` r
# Análise rápida dos resultados para identificar padrões de falhas (neuronios vazios, erros) e distribuição de QE
empty_configs <- results_df |>
    dplyr::filter(empty_neurons > 0) |>
    dplyr::count(dataset, neighborhood, topology, empty_neurons)

error_configs <- results_df |>
    dplyr::filter(is.null(model)) |>
    dplyr::count(dataset, neighborhood, topology)

qe_distribution <- results_df |>
    dplyr::filter(!is.na(qe_val)) |>
    dplyr::group_by(dataset, neighborhood, fold) |>
    dplyr::summarise(
        n = dplyr::n(),
        qe_min = min(qe_val),
        qe_mediana = median(qe_val),
        qe_mean = mean(qe_val),
        qe_max = max(qe_val),
        .groups = "drop"
    )

cat("=== CONFIGURACOES COM NEURONIOS VAZIOS ===\n")
```

    === CONFIGURACOES COM NEURONIOS VAZIOS ===

``` r
print(empty_configs)
```

      dataset neighborhood    topology empty_neurons n
    1   scene       bubble   hexagonal             1 2
    2   scene       bubble rectangular             1 1

``` r
cat("=== CONFIGURACOES COM ERROS (VALORES NA) ===\n")
```

    === CONFIGURACOES COM ERROS (VALORES NA) ===

``` r
print(error_configs)
```

    [1] dataset      neighborhood topology     n           
    <0 rows> (or 0-length row.names)

``` r
cat("\n=== DISTRIBUICAO DE QE POR DATASET E VIZINHANCA ===\n")
```


    === DISTRIBUICAO DE QE POR DATASET E VIZINHANCA ===

``` r
print(qe_distribution)
```

    # A tibble: 12 × 8
       dataset  neighborhood  fold     n qe_min qe_mediana qe_mean qe_max
       <chr>    <fct>        <int> <int>  <dbl>      <dbl>   <dbl>  <dbl>
     1 emotions bubble           1    48  0.458      0.467   0.477  0.522
     2 emotions bubble           2    48  0.456      0.463   0.470  0.495
     3 emotions bubble           3    48  0.458      0.468   0.476  0.525
     4 emotions gaussian         1    48  0.544      0.574   0.572  0.604
     5 emotions gaussian         2    48  0.533      0.571   0.566  0.622
     6 emotions gaussian         3    48  0.534      0.571   0.566  0.612
     7 scene    bubble           1    48  0.358      0.372   0.382  0.555
     8 scene    bubble           2    48  0.358      0.380   0.386  0.538
     9 scene    bubble           3    48  0.358      0.373   0.379  0.529
    10 scene    gaussian         1    48  0.426      0.446   0.444  0.470
    11 scene    gaussian         2    48  0.423      0.446   0.446  0.472
    12 scene    gaussian         3    48  0.422      0.445   0.445  0.477

``` r
write.csv(qe_distribution, file.path(OUTPUT_PATH, "qe_distribution.csv"), row.names = FALSE)



# Para cada dataset+vizinhança, testa se radius, rlen, alpha_start, alpha_end movem o QE
param_effect_tests <- results_df |>
    dplyr::filter(!is.na(qe_val), empty_neurons == 0) |>
    dplyr::group_by(dataset, neighborhood) |>
    dplyr::summarise(
        p_radius     = kruskal.test(qe_val ~ factor(radius),     data = dplyr::pick(everything()))$p.value,
        p_rlen       = kruskal.test(qe_val ~ factor(rlen),       data = dplyr::pick(everything()))$p.value,
        p_alpha_start = kruskal.test(qe_val ~ factor(alpha_start), data = dplyr::pick(everything()))$p.value,
        p_alpha_end   = kruskal.test(qe_val ~ factor(alpha_end),   data = dplyr::pick(everything()))$p.value,
        .groups = "drop"
    )

cat("\n=== TESTES DE EFEITO DOS PARAMETROS ===\n")
```


    === TESTES DE EFEITO DOS PARAMETROS ===

``` r
print(param_effect_tests)
```

    # A tibble: 4 × 6
      dataset  neighborhood p_radius p_rlen p_alpha_start p_alpha_end
      <chr>    <fct>           <dbl>  <dbl>         <dbl>       <dbl>
    1 emotions bubble        0.139   0.0217        0.183       0.814 
    2 emotions gaussian      0.920   0.165         0.0140      0.959 
    3 scene    bubble        0.00240 0.193         0.665       0.0818
    4 scene    gaussian      0.270   0.470         0.0260      0.242 

``` r
# p > 0.05 → parâmetro não está movendo QE de forma confiável (candidato a dredging)
```

## Seleção dos Melhores Parâmetros

``` r
# Estratégia:
# 1. Remove configurações com falhas
# 2. Exige que a configuração tenha rodado nos 3 folds (estabilidade)
# 3. Define o minimo e calcula QE médio entre os folds
# 4. Seleciona o melhor por (dataset, vizinhança) — topologia também é fixada aqui
#    pois todos os 3 folds devem usar a mesma configuração
best_params_mean <- results_df |>
    dplyr::filter(!is.na(qe_val)) |>
    dplyr::group_by(dataset, neighborhood, topology, rlen, radius, alpha_start, alpha_end) |>
    dplyr::reframe(
        mean_qe  = mean(qe_val),
        sd_qe    = sd(qe_val),
        n_obs    = dplyr::n(),
        n_folds  = dplyr::n_distinct(fold)
    ) |>
    dplyr::filter(n_folds == 3) |>
    dplyr::group_by(dataset, neighborhood) |>
    dplyr::slice_min(mean_qe, n = 1, with_ties = FALSE) |>
    dplyr::ungroup()

best_params_min <- results_df |>
    dplyr::filter(!is.na(qe_val)) |>
    dplyr::group_by(dataset, neighborhood, topology, rlen, radius, alpha_start, alpha_end) |>
    dplyr::reframe(
        min_qe  = min(qe_val),
        mean_qe  = mean(qe_val),
        sd_qe    = sd(qe_val),
        n_obs    = dplyr::n(),
        n_folds  = dplyr::n_distinct(fold)
    ) |>
    dplyr::filter(n_folds == 3) |>
    dplyr::group_by(dataset, neighborhood) |>
    dplyr::slice_min(min_qe, n = 1, with_ties = FALSE) |>
    dplyr::ungroup()

cat("=== MELHORES PARAMETROS POR DATASET E VIZINHANCA ===\n")
```

    === MELHORES PARAMETROS POR DATASET E VIZINHANCA ===

``` r
print(best_params_mean)
```

    # A tibble: 4 × 11
      dataset  neighborhood topology     rlen radius alpha_start alpha_end mean_qe
      <chr>    <fct>        <chr>       <int>  <dbl>       <dbl>     <dbl>   <dbl>
    1 emotions bubble       rectangular  1000    1.5        0.1      0.005   0.459
    2 emotions gaussian     hexagonal    1000    1          0.1      0.005   0.541
    3 scene    bubble       rectangular   500    1.5        0.05     0.005   0.364
    4 scene    gaussian     hexagonal     500    1.5        0.1      0.01    0.428
    # ℹ 3 more variables: sd_qe <dbl>, n_obs <int>, n_folds <int>

``` r
print(best_params_min)
```

    # A tibble: 4 × 12
      dataset  neighborhood topology     rlen radius alpha_start alpha_end min_qe
      <chr>    <fct>        <chr>       <int>  <dbl>       <dbl>     <dbl>  <dbl>
    1 emotions bubble       hexagonal    1000    1.5        0.05     0.005  0.456
    2 emotions gaussian     hexagonal     500    1.5        0.1      0.005  0.533
    3 scene    bubble       rectangular   500    1.5        0.05     0.005  0.358
    4 scene    gaussian     hexagonal    1000    1.5        0.1      0.01   0.422
    # ℹ 4 more variables: mean_qe <dbl>, sd_qe <dbl>, n_obs <int>, n_folds <int>

``` r
best_params_csv <- file.path(OUTPUT_PATH, "best_params.csv")
write.csv(best_params_mean, best_params_csv, row.names = FALSE)
cat(sprintf("\nMelhores parametros (média) salvos em: %s\n", best_params_csv))
```


    Melhores parametros (média) salvos em: E:/git-tcc/tcc-final/outputs/gridsearch/best_params.csv

``` r
best_params_csv_min <- file.path(OUTPUT_PATH, "best_params_min.csv")
write.csv(best_params_min, best_params_csv_min, row.names = FALSE)
cat(sprintf("Melhores parametros (mínimo absoluto) salvos em: %s\n", best_params_csv_min))
```

    Melhores parametros (mínimo absoluto) salvos em: E:/git-tcc/tcc-final/outputs/gridsearch/best_params_min.csv

``` r
best_params_mean <- best_params_mean |>
    dplyr::mutate(
        config = sprintf("%s|%s", neighborhood, topology),
        param = sprintf("%d %.1f (%.2f,%.3f)", rlen, radius, alpha_start, alpha_end),
        cv = sd_qe / mean_qe, # coeficiente de variação
        instavel = cv > 0.05 # TRUE = resultado pode ser sorte
    )
cat("\n=== MELHORES PARAMETROS COM CV E INSTABILIDADE ===\n")
```


    === MELHORES PARAMETROS COM CV E INSTABILIDADE ===

``` r
print(best_params_mean |> dplyr::select(config, param, mean_qe, sd_qe, cv, instavel))
```

    # A tibble: 4 × 6
      config             param                 mean_qe   sd_qe      cv instavel
      <chr>              <chr>                   <dbl>   <dbl>   <dbl> <lgl>   
    1 bubble|rectangular 1000 1.5 (0.10,0.005)   0.459 0.00146 0.00319 FALSE   
    2 gaussian|hexagonal 1000 1.0 (0.10,0.005)   0.541 0.00583 0.0108  FALSE   
    3 bubble|rectangular 500 1.5 (0.05,0.005)    0.364 0.00539 0.0148  FALSE   
    4 gaussian|hexagonal 500 1.5 (0.10,0.010)    0.428 0.00173 0.00403 FALSE   

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
    [1] stats     graphics  grDevices utils     datasets  methods   base     

    other attached packages:
     [1] kohonen_3.0.12  lubridate_1.9.4 forcats_1.0.1   stringr_1.6.0  
     [5] dplyr_1.1.4     purrr_1.2.1     readr_2.1.6     tidyr_1.3.2    
     [9] tibble_3.3.1    ggplot2_4.0.1   tidyverse_2.0.0

    loaded via a namespace (and not attached):
     [1] gtable_0.3.6       jsonlite_2.0.0     compiler_4.4.3     Rcpp_1.1.1        
     [5] tidyselect_1.2.1   scales_1.4.0       yaml_2.3.12        fastmap_1.2.0     
     [9] R6_2.6.1           generics_0.1.4     knitr_1.51         pillar_1.11.1     
    [13] RColorBrewer_1.1-3 tzdb_0.5.0         rlang_1.1.7        utf8_1.2.6        
    [17] stringi_1.8.7      xfun_0.56          S7_0.2.1           otel_0.2.0        
    [21] timechange_0.3.0   cli_3.6.5          withr_3.0.2        magrittr_2.0.4    
    [25] digest_0.6.39      grid_4.4.3         hms_1.1.4          lifecycle_1.0.5   
    [29] vctrs_0.7.0        evaluate_1.0.5     glue_1.8.0         farver_2.1.2      
    [33] rmarkdown_2.30     tools_4.4.3        pkgconfig_2.0.3    htmltools_0.5.9   

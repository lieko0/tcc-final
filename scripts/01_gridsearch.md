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

    [    0.1s]    1/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.58947
    [    0.1s]    2/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.60297
    [    0.2s]    3/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.58452
    [    0.2s]    4/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.57278
    [    0.2s]    5/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.59292
    [    0.3s]    6/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.60109
    [    0.3s]    7/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.58405
    [    0.3s]    8/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.58106
    [    0.3s]    9/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57628
    [    0.4s]   10/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.60340
    [    0.4s]   11/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.58918
    [    0.4s]   12/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.57492
    [    0.5s]   13/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58126
    [    0.5s]   14/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.60385
    [    0.5s]   15/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57923
    [    0.5s]   16/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.60119
    [    0.6s]   17/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.58631
    [    0.6s]   18/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59192
    [    0.7s]   19/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.58184
    [    0.7s]   20/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.58565
    [    0.7s]   21/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.58055
    [    0.7s]   22/576 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.58727
    [    0.8s]   23/576 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.58196
    [    0.8s]   24/576 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.57241
    [    0.8s]   25/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.59885
    [    0.9s]   26/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.58747
    [    0.9s]   27/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.56442
    [    0.9s]   28/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.57297
    [    1.0s]   29/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.55395
    [    1.0s]   30/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.54516
    [    1.0s]   31/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.55897
    [    1.0s]   32/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.54414
    [    1.1s]   33/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.54744
    [    1.1s]   34/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.54472
    [    1.1s]   35/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.55074
    [    1.2s]   36/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.55720
    [    1.2s]   37/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56643
    [    1.2s]   38/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55016
    [    1.3s]   39/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.54718
    [    1.3s]   40/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.56261
    [    1.3s]   41/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.56228
    [    1.4s]   42/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54419
    [    1.4s]   43/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.55823
    [    1.4s]   44/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.55115
    [    1.5s]   45/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.54507
    [    1.5s]   46/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54615
    [    1.5s]   47/576 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.55914
    [    1.6s]   48/576 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54736
    [    1.6s]   49/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.49540
    [    1.6s]   50/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.50326
    [    1.6s]   51/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46588
    [    1.6s]   52/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.50817
    [    1.6s]   53/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.48502
    [    1.7s]   54/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45848
    [    1.7s]   55/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46714
    [    1.7s]   56/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.48117
    [    1.7s]   57/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45838
    [    1.7s]   58/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.48262
    [    1.7s]   59/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45927
    [    1.8s]   60/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45895
    [    1.8s]   61/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.50928
    [    1.8s]   62/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49606
    [    1.8s]   63/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46671
    [    1.8s]   64/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48128
    [    1.8s]   65/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46628
    [    1.9s]   66/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45904
    [    1.9s]   67/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45974
    [    1.9s]   68/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45997
    [    1.9s]   69/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46405
    [    1.9s]   70/576 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.49608
    [    1.9s]   71/576 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46150
    [    2.0s]   72/576 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45905
    [    2.0s]   73/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.49530
    [    2.0s]   74/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.48157
    [    2.0s]   75/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.50848
    [    2.0s]   76/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.48383
    [    2.0s]   77/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.50223
    [    2.1s]   78/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45852
    [    2.1s]   79/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45935
    [    2.1s]   80/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46619
    [    2.1s]   81/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45839
    [    2.1s]   82/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46685
    [    2.1s]   83/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46006
    [    2.2s]   84/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46048
    [    2.2s]   85/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46007
    [    2.2s]   86/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.50498
    [    2.2s]   87/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46586
    [    2.2s]   88/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.52202
    [    2.2s]   89/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49640
    [    2.2s]   90/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48562
    [    2.2s]   91/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46887
    [    2.3s]   92/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.48211
    [    2.3s]   93/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.48336
    [    2.3s]   94/576 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45960
    [    2.3s]   95/576 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.48462
    [    2.3s]   96/576 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46684
    [    2.4s]   97/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44836
    [    2.5s]   98/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44930
    [    2.7s]   99/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45709
    [    2.7s]  100/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44819
    [    2.8s]  101/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44831
    [    3.0s]  102/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45070
    [    3.1s]  103/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45277
    [    3.2s]  104/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43875
    [    3.4s]  105/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45160
    [    3.4s]  106/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45195
    [    3.5s]  107/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.44767
    [    3.7s]  108/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.44869
    [    3.8s]  109/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44732
    [    3.9s]  110/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46105
    [    4.1s]  111/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44744
    [    4.1s]  112/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44964
    [    4.2s]  113/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45004
    [    4.4s]  114/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46817
    [    4.5s]  115/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44672
    [    4.6s]  116/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45492
    [    4.8s]  117/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45218
    [    4.8s]  118/576 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45225
    [    4.9s]  119/576 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.44157
    [    5.1s]  120/576 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45487
    [    5.2s]  121/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44031
    [    5.3s]  122/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.43036
    [    5.4s]  123/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.43340
    [    5.5s]  124/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43715
    [    5.6s]  125/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.42973
    [    5.8s]  126/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45373
    [    5.8s]  127/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43472
    [    5.9s]  128/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43822
    [    6.1s]  129/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.42607
    [    6.2s]  130/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.44552
    [    6.3s]  131/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.43289
    [    6.5s]  132/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.42751
    [    6.5s]  133/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46975
    [    6.6s]  134/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43538
    [    6.8s]  135/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43567
    [    6.8s]  136/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43598
    [    7.0s]  137/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44366
    [    7.2s]  138/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43613
    [    7.2s]  139/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44314
    [    7.3s]  140/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44137
    [    7.5s]  141/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.43893
    [    7.5s]  142/576 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.43481
    [    7.6s]  143/576 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.42931
    [    7.8s]  144/576 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.42878
    [    7.9s]  145/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37576
    [    7.9s]  146/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37856
    [    8.1s]  147/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.50918
    [    8.1s]  148/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37207
    [    8.1s]  149/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36682
    [    8.3s]  150/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36634
    [    8.3s]  151/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.37429
    [    8.3s]  152/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.36056
    [    8.4s]  153/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.37288
    [    8.5s]  154/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.36554
    [    8.6s]  155/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.36212
    [    8.7s]  156/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.35801
    [    8.7s]  157/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36986
    [    8.8s]  158/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37825
    [    8.9s]  159/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38063
    [    8.9s]  160/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36400
    [    9.0s]  161/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36333
    [    9.1s]  162/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37225
    [    9.2s]  163/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38164
    [    9.2s]  164/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38065
    [    9.4s]  165/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.52771
    [    9.4s]  166/576 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.35828
    [    9.4s]  167/576 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36716
    [    9.6s]  168/576 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.38400
    [    9.6s]  169/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.36890
    [    9.7s]  170/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.36918
    [    9.8s]  171/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.39010
    [    9.8s]  172/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37345
    [    9.9s]  173/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36211
    [   10.0s]  174/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37152
    [   10.1s]  175/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.36007
    [   10.1s]  176/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.36906
    [   10.2s]  177/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.38073
    [   10.2s]  178/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.37845
    [   10.3s]  179/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.35816
    [   10.4s]  180/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.38037
    [   10.5s]  181/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37157
    [   10.5s]  182/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=1 QE=0.55475
    [   10.6s]  183/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37179
    [   10.7s]  184/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38329
    [   10.7s]  185/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.35845
    [   10.9s]  186/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37937
    [   10.9s]  187/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38162
    [   10.9s]  188/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38080
    [   11.1s]  189/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37924
    [   11.1s]  190/576 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36649
    [   11.2s]  191/576 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37222
    [   11.3s]  192/576 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37340
    [   11.3s]  193/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.58783
    [   11.3s]  194/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.58717
    [   11.4s]  195/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.57961
    [   11.4s]  196/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.62206
    [   11.4s]  197/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.58970
    [   11.5s]  198/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.58897
    [   11.5s]  199/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57741
    [   11.5s]  200/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57840
    [   11.6s]  201/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57468
    [   11.6s]  202/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.58150
    [   11.6s]  203/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.58605
    [   11.7s]  204/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.57168
    [   11.7s]  205/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57016
    [   11.7s]  206/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57359
    [   11.8s]  207/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57853
    [   11.8s]  208/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.57403
    [   11.8s]  209/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.58565
    [   11.9s]  210/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.57672
    [   11.9s]  211/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.58284
    [   11.9s]  212/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.57120
    [   11.9s]  213/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.57443
    [   11.9s]  214/576 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.58993
    [   12.0s]  215/576 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.57546
    [   12.0s]  216/576 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.57805
    [   12.1s]  217/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.57565
    [   12.1s]  218/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.56038
    [   12.1s]  219/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.56207
    [   12.1s]  220/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.53614
    [   12.2s]  221/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.54708
    [   12.2s]  222/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.54146
    [   12.2s]  223/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.55961
    [   12.3s]  224/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.53923
    [   12.4s]  225/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.53975
    [   12.4s]  226/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.54333
    [   12.4s]  227/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.53286
    [   12.5s]  228/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.53427
    [   12.5s]  229/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56479
    [   12.5s]  230/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57546
    [   12.6s]  231/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.54596
    [   12.6s]  232/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54433
    [   12.6s]  233/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54963
    [   12.6s]  234/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54298
    [   12.7s]  235/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.55176
    [   12.7s]  236/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.56187
    [   12.7s]  237/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.55869
    [   12.7s]  238/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54079
    [   12.8s]  239/576 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.55164
    [   12.8s]  240/576 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54293
    [   12.8s]  241/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46272
    [   12.8s]  242/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46212
    [   12.9s]  243/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.47629
    [   12.9s]  244/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.46152
    [   12.9s]  245/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.47955
    [   12.9s]  246/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.48169
    [   12.9s]  247/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.47626
    [   12.9s]  248/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46145
    [   12.9s]  249/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.47541
    [   13.0s]  250/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.49445
    [   13.0s]  251/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46161
    [   13.0s]  252/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46100
    [   13.0s]  253/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.47767
    [   13.1s]  254/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46417
    [   13.1s]  255/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46186
    [   13.1s]  256/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.47999
    [   13.1s]  257/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46101
    [   13.1s]  258/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48118
    [   13.1s]  259/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46407
    [   13.1s]  260/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46152
    [   13.2s]  261/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.48137
    [   13.2s]  262/576 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46241
    [   13.2s]  263/576 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45889
    [   13.2s]  264/576 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46334
    [   13.2s]  265/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46173
    [   13.2s]  266/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45783
    [   13.2s]  267/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46065
    [   13.3s]  268/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.49261
    [   13.3s]  269/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.47930
    [   13.3s]  270/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45636
    [   13.3s]  271/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.47884
    [   13.3s]  272/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.47697
    [   13.3s]  273/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45838
    [   13.4s]  274/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.49371
    [   13.4s]  275/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.47951
    [   13.4s]  276/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.49274
    [   13.4s]  277/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46209
    [   13.4s]  278/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46237
    [   13.4s]  279/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45729
    [   13.4s]  280/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46061
    [   13.5s]  281/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45816
    [   13.5s]  282/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48127
    [   13.5s]  283/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46162
    [   13.5s]  284/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46470
    [   13.6s]  285/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46003
    [   13.6s]  286/576 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.49498
    [   13.6s]  287/576 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.49267
    [   13.6s]  288/576 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45930
    [   13.6s]  289/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44901
    [   13.7s]  290/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45708
    [   13.9s]  291/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45426
    [   13.9s]  292/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44182
    [   14.1s]  293/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44797
    [   14.3s]  294/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.47004
    [   14.3s]  295/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45850
    [   14.4s]  296/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45733
    [   14.6s]  297/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46159
    [   14.6s]  298/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45439
    [   14.8s]  299/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45444
    [   14.9s]  300/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45169
    [   15.0s]  301/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44691
    [   15.1s]  302/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.47204
    [   15.3s]  303/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45264
    [   15.3s]  304/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45962
    [   15.4s]  305/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46174
    [   15.6s]  306/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45483
    [   15.7s]  307/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46385
    [   15.8s]  308/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45431
    [   16.0s]  309/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44117
    [   16.0s]  310/576 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45062
    [   16.1s]  311/576 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.44256
    [   16.3s]  312/576 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.44022
    [   16.4s]  313/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.43844
    [   16.5s]  314/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45166
    [   16.7s]  315/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44162
    [   16.7s]  316/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44574
    [   16.8s]  317/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43046
    [   17.0s]  318/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43983
    [   17.0s]  319/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43707
    [   17.1s]  320/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.44407
    [   17.3s]  321/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43412
    [   17.4s]  322/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.42502
    [   17.5s]  323/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.43545
    [   17.7s]  324/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.43412
    [   17.7s]  325/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46532
    [   17.8s]  326/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43957
    [   18.0s]  327/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.42339
    [   18.0s]  328/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44580
    [   18.2s]  329/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43825
    [   18.4s]  330/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43318
    [   18.4s]  331/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.42505
    [   18.5s]  332/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.43066
    [   18.7s]  333/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44182
    [   18.7s]  334/576 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45688
    [   18.8s]  335/576 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.42820
    [   19.0s]  336/576 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.43478
    [   19.0s]  337/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.38200
    [   19.1s]  338/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.39360
    [   19.2s]  339/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37173
    [   19.2s]  340/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37052
    [   19.3s]  341/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36744
    [   19.4s]  342/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.35893
    [   19.5s]  343/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.52640
    [   19.6s]  344/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.38324
    [   19.6s]  345/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.36996
    [   19.7s]  346/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.38796
    [   19.8s]  347/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.35841
    [   19.9s]  348/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.39125
    [   19.9s]  349/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.53844
    [   20.0s]  350/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38912
    [   20.1s]  351/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38772
    [   20.1s]  352/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.39514
    [   20.2s]  353/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36616
    [   20.3s]  354/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37358
    [   20.3s]  355/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38203
    [   20.4s]  356/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.36087
    [   20.5s]  357/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37407
    [   20.6s]  358/576 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36036
    [   20.6s]  359/576 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36317
    [   20.8s]  360/576 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36093
    [   20.8s]  361/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.36110
    [   20.8s]  362/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.38253
    [   21.0s]  363/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37072
    [   21.0s]  364/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.38373
    [   21.0s]  365/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37044
    [   21.2s]  366/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.38390
    [   21.2s]  367/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=1 QE=0.53100
    [   21.2s]  368/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.38152
    [   21.4s]  369/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.37187
    [   21.4s]  370/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.38234
    [   21.4s]  371/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.36611
    [   21.6s]  372/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.38102
    [   21.6s]  373/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36133
    [   21.7s]  374/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.39194
    [   21.8s]  375/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.39147
    [   21.8s]  376/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38359
    [   21.9s]  377/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38200
    [   22.0s]  378/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38399
    [   22.0s]  379/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37402
    [   22.1s]  380/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37209
    [   22.2s]  381/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.36244
    [   22.2s]  382/576 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37951
    [   22.3s]  383/576 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.38070
    [   22.4s]  384/576 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37020
    [   22.4s]  385/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.57692
    [   22.5s]  386/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.57682
    [   22.5s]  387/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.57684
    [   22.6s]  388/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.59517
    [   22.6s]  389/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.58286
    [   22.6s]  390/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.59943
    [   22.6s]  391/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57228
    [   22.7s]  392/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57371
    [   22.7s]  393/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.57371
    [   22.7s]  394/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.58683
    [   22.8s]  395/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.61164
    [   22.8s]  396/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.57337
    [   22.8s]  397/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58162
    [   22.8s]  398/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.60969
    [   22.9s]  399/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57353
    [   22.9s]  400/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.60252
    [   22.9s]  401/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.60017
    [   23.0s]  402/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.57500
    [   23.0s]  403/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.56516
    [   23.0s]  404/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.57374
    [   23.1s]  405/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.57305
    [   23.1s]  406/576 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.60688
    [   23.1s]  407/576 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.59848
    [   23.2s]  408/576 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.58503
    [   23.2s]  409/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.54509
    [   23.2s]  410/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.54065
    [   23.3s]  411/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.54448
    [   23.3s]  412/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.54074
    [   23.3s]  413/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.55618
    [   23.4s]  414/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.55621
    [   23.4s]  415/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.54214
    [   23.4s]  416/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.54400
    [   23.4s]  417/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.53601
    [   23.4s]  418/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.55079
    [   23.5s]  419/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.54335
    [   23.5s]  420/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.53385
    [   23.6s]  421/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58595
    [   23.6s]  422/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56973
    [   23.6s]  423/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55346
    [   23.6s]  424/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.55379
    [   23.7s]  425/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54043
    [   23.7s]  426/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.55569
    [   23.7s]  427/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.54339
    [   23.8s]  428/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.54222
    [   23.8s]  429/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.54582
    [   23.8s]  430/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54012
    [   23.8s]  431/576 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.54227
    [   23.9s]  432/576 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.53438
    [   23.9s]  433/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46484
    [   23.9s]  434/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.46004
    [   23.9s]  435/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.49798
    [   24.0s]  436/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.48018
    [   24.0s]  437/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.46503
    [   24.0s]  438/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45904
    [   24.0s]  439/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46196
    [   24.0s]  440/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45907
    [   24.0s]  441/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46710
    [   24.0s]  442/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.48435
    [   24.1s]  443/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.48304
    [   24.1s]  444/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45817
    [   24.1s]  445/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45849
    [   24.1s]  446/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49895
    [   24.1s]  447/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46038
    [   24.1s]  448/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48237
    [   24.2s]  449/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48482
    [   24.2s]  450/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46012
    [   24.2s]  451/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.48517
    [   24.2s]  452/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46190
    [   24.2s]  453/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46033
    [   24.2s]  454/576 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45857
    [   24.2s]  455/576 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.48100
    [   24.3s]  456/576 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.48534
    [   24.3s]  457/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.48216
    [   24.3s]  458/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45845
    [   24.3s]  459/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45932
    [   24.3s]  460/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.46066
    [   24.4s]  461/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.49750
    [   24.4s]  462/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.49901
    [   24.4s]  463/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.49918
    [   24.4s]  464/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45983
    [   24.4s]  465/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.46639
    [   24.4s]  466/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.49713
    [   24.4s]  467/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.46532
    [   24.5s]  468/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.49795
    [   24.5s]  469/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48124
    [   24.5s]  470/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48211
    [   24.5s]  471/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46110
    [   24.5s]  472/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46012
    [   24.5s]  473/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49701
    [   24.6s]  474/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.52483
    [   24.6s]  475/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.48083
    [   24.6s]  476/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46260
    [   24.6s]  477/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.46856
    [   24.6s]  478/576 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.49802
    [   24.6s]  479/576 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.49879
    [   24.7s]  480/576 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46199
    [   24.7s]  481/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45639
    [   24.8s]  482/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45193
    [   25.0s]  483/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.45391
    [   25.0s]  484/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.44992
    [   25.1s]  485/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45710
    [   25.3s]  486/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.45357
    [   25.4s]  487/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.44768
    [   25.5s]  488/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.44959
    [   25.7s]  489/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.45727
    [   25.7s]  490/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45201
    [   25.8s]  491/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.45216
    [   26.0s]  492/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.44040
    [   26.1s]  493/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.47718
    [   26.2s]  494/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45860
    [   26.4s]  495/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45649
    [   26.4s]  496/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46383
    [   26.5s]  497/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45315
    [   26.7s]  498/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45257
    [   26.8s]  499/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44740
    [   26.8s]  500/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.45786
    [   27.0s]  501/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44049
    [   27.1s]  502/576 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.45616
    [   27.2s]  503/576 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.46670
    [   27.4s]  504/576 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.44359
    [   27.4s]  505/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.42722
    [   27.5s]  506/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.44406
    [   27.7s]  507/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.43043
    [   27.8s]  508/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43985
    [   27.9s]  509/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43463
    [   28.1s]  510/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.43389
    [   28.1s]  511/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.42844
    [   28.2s]  512/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.44337
    [   28.4s]  513/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.43196
    [   28.4s]  514/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.42947
    [   28.6s]  515/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.42681
    [   28.8s]  516/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.44031
    [   28.8s]  517/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44275
    [   28.9s]  518/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45918
    [   29.1s]  519/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44045
    [   29.1s]  520/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45676
    [   29.2s]  521/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43897
    [   29.4s]  522/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43213
    [   29.5s]  523/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.43341
    [   29.6s]  524/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44555
    [   29.8s]  525/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.44417
    [   29.8s]  526/576 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.43478
    [   29.9s]  527/576 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.42592
    [   30.1s]  528/576 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.42247
    [   30.1s]  529/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37898
    [   30.2s]  530/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.35849
    [   30.3s]  531/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.38192
    [   30.3s]  532/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37458
    [   30.4s]  533/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.35780
    [   30.5s]  534/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36762
    [   30.5s]  535/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.35846
    [   30.6s]  536/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=1 QE=0.52915
    [   30.7s]  537/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.38084
    [   30.7s]  538/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.37259
    [   30.8s]  539/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.37216
    [   30.9s]  540/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.35842
    [   31.0s]  541/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36938
    [   31.0s]  542/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.39398
    [   31.1s]  543/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37276
    [   31.1s]  544/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36413
    [   31.2s]  545/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36332
    [   31.4s]  546/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38365
    [   31.4s]  547/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37338
    [   31.4s]  548/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38068
    [   31.6s]  549/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.50981
    [   31.6s]  550/576 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37852
    [   31.7s]  551/576 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36786
    [   31.8s]  552/576 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37016
    [   31.8s]  553/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37101
    [   31.9s]  554/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37042
    [   32.0s]  555/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.005) | empty=0 QE=0.37141
    [   32.0s]  556/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.36362
    [   32.0s]  557/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37316
    [   32.2s]  558/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.005) | empty=0 QE=0.37020
    [   32.2s]  559/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.37162
    [   32.3s]  560/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.38410
    [   32.4s]  561/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.005) | empty=0 QE=0.37369
    [   32.4s]  562/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.36318
    [   32.4s]  563/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.36868
    [   32.6s]  564/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.005) | empty=0 QE=0.38222
    [   32.6s]  565/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38185
    [   32.7s]  566/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37301
    [   32.8s]  567/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36130
    [   32.8s]  568/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38050
    [   32.8s]  569/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38350
    [   33.0s]  570/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36617
    [   33.0s]  571/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38273
    [   33.0s]  572/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.38540
    [   33.1s]  573/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.10,0.010) | empty=0 QE=0.37364
    [   33.2s]  574/576 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.36279
    [   33.2s]  575/576 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.37096
    [   33.4s]  576/576 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.10,0.010) | empty=0 QE=0.39440

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

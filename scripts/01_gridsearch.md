# Script 1 — Grid Search


## Configuração

``` r
library(tidyverse)
library(kohonen)

BASE_PATH      <- "E:/git-tcc/tcc-final/data"
OUTPUT_PATH    <- "E:/git-tcc/tcc-final/outputs/gridsearch"
SEED           <- 1234L
# Conforme o PDF: apenas emotions e scene
TARGET_DATASETS <- c("emotions", "scene")

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
    cat(sprintf("  %-10s — %d atributos, %d rotulos\n",
                ds, length(cfg$attr_cols), length(cfg$label_cols)))
}
```

      emotions   — 72 atributos, 6 rotulos
      scene      — 294 atributos, 6 rotulos

## Definição das Grades SOM (2×2 apenas)

``` r
# Conforme o PDF: apenas grade 2x2, 2 vizinhancas x 2 topologias = 4 grids
grids <- list(
    gaussian_rectangular = kohonen::somgrid(2, 2, "rectangular", neighbourhood.fct = "gaussian"),
    gaussian_hexagonal   = kohonen::somgrid(2, 2, "hexagonal",   neighbourhood.fct = "gaussian"),
    bubble_rectangular   = kohonen::somgrid(2, 2, "rectangular", neighbourhood.fct = "bubble"),
    bubble_hexagonal     = kohonen::somgrid(2, 2, "hexagonal",   neighbourhood.fct = "bubble")
)

cat("Grades definidas:\n")
```

    Grades definidas:

``` r
for (name in names(grids)) {
    g <- grids[[name]]
    cat(sprintf("  %-28s — %dx%d | topo=%-15s | viz=%s\n",
                name, g$xdim, g$ydim, g$topo, g$neighbourhood.fct))
}
```

      gaussian_rectangular         — 2x2 | topo=rectangular     | viz=gaussian
      gaussian_hexagonal           — 2x2 | topo=hexagonal       | viz=gaussian
      bubble_rectangular           — 2x2 | topo=rectangular     | viz=bubble
      bubble_hexagonal             — 2x2 | topo=hexagonal       | viz=bubble

## Grade de Parâmetros

``` r
param_grid <- expand.grid(
    rlen        = c(100L, 200L, 500L), # default: 100L
    radius      = c(0.5, 1.0, 1.5), # default: 1.0
    alpha_start = c(0.05), # default: 0.05
    alpha_end   = c(0.01), # default: 0.01
    grid_name   = names(grids),
    dataset     = TARGET_DATASETS,
    fold        = 1:3,
    stringsAsFactors = FALSE
)

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

    Total de combinacoes: 216
      (4 grids x 2 datasets x 3 folds x 9 combinacoes de params)

## Execução do Grid Search

``` r
results_list <- vector("list", nrow(param_grid))
start_time   <- proc.time()

for (i in seq_len(nrow(param_grid))) {
    p    <- param_grid[i, ]
    grid <- grids[[p$grid_name]]

    Y_tr <- as.matrix(datasets[[p$dataset]][[paste0("fold", p$fold)]][["tr"]]$labels)
    Y_vl <- as.matrix(datasets[[p$dataset]][[paste0("fold", p$fold)]][["vl"]]$labels)

    model <- tryCatch({
        kohonen::supersom(
            data      = list(Y = Y_tr),
            grid      = grid,
            rlen      = p$rlen,
            radius    = p$radius,
            alpha     = c(p$alpha_start, p$alpha_end),
            keep.data = TRUE
        )
    }, error = function(e) {
        message("Erro (i=", i, "): ", e$message)
        NULL
    })

    if (is.null(model)) {
        qe_vl   <- NA_real_
        empty_n <- NA_integer_
    } else {
        grid_size <- grid$xdim * grid$ydim
        empty_n   <- grid_size - length(unique(model$unit.classif))
        qe_vl     <- mean(kohonen::map(model, newdata = list(Y = Y_vl))$distances)
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

    [    0.1s]    1/216 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.59277
    [    0.1s]    2/216 | emotions  fold1 | gaussian_rectangular     | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.59646
    [    0.1s]    3/216 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.59886
    [    0.1s]    4/216 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59719
    [    0.1s]    5/216 | emotions  fold1 | gaussian_rectangular     | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59453
    [    0.2s]    6/216 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59222
    [    0.2s]    7/216 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.57219
    [    0.2s]    8/216 | emotions  fold1 | gaussian_rectangular     | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.58860
    [    0.2s]    9/216 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59640
    [    0.2s]   10/216 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.57258
    [    0.3s]   11/216 | emotions  fold1 | gaussian_hexagonal       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.54507
    [    0.3s]   12/216 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.58050
    [    0.3s]   13/216 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56634
    [    0.3s]   14/216 | emotions  fold1 | gaussian_hexagonal       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.54969
    [    0.3s]   15/216 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.54365
    [    0.3s]   16/216 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.58818
    [    0.4s]   17/216 | emotions  fold1 | gaussian_hexagonal       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.55548
    [    0.4s]   18/216 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.56015
    [    0.4s]   19/216 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.48118
    [    0.4s]   20/216 | emotions  fold1 | bubble_rectangular       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45981
    [    0.4s]   21/216 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46822
    [    0.4s]   22/216 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48521
    [    0.4s]   23/216 | emotions  fold1 | bubble_rectangular       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46605
    [    0.5s]   24/216 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46146
    [    0.5s]   25/216 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48435
    [    0.5s]   26/216 | emotions  fold1 | bubble_rectangular       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48538
    [    0.5s]   27/216 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.50865
    [    0.5s]   28/216 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.48219
    [    0.5s]   29/216 | emotions  fold1 | bubble_hexagonal         | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.49634
    [    0.5s]   30/216 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.51570
    [    0.5s]   31/216 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48390
    [    0.5s]   32/216 | emotions  fold1 | bubble_hexagonal         | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49700
    [    0.6s]   33/216 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45981
    [    0.6s]   34/216 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49839
    [    0.6s]   35/216 | emotions  fold1 | bubble_hexagonal         | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48608
    [    0.6s]   36/216 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49518
    [    0.6s]   37/216 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46317
    [    0.7s]   38/216 | scene     fold1 | gaussian_rectangular     | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45736
    [    0.8s]   39/216 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46225
    [    0.9s]   40/216 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46149
    [    0.9s]   41/216 | scene     fold1 | gaussian_rectangular     | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44585
    [    1.0s]   42/216 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45890
    [    1.0s]   43/216 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44918
    [    1.1s]   44/216 | scene     fold1 | gaussian_rectangular     | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44721
    [    1.2s]   45/216 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45651
    [    1.2s]   46/216 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45560
    [    1.3s]   47/216 | scene     fold1 | gaussian_hexagonal       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.44509
    [    1.4s]   48/216 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.44531
    [    1.4s]   49/216 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43953
    [    1.5s]   50/216 | scene     fold1 | gaussian_hexagonal       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.42857
    [    1.6s]   51/216 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43363
    [    1.6s]   52/216 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43683
    [    1.7s]   53/216 | scene     fold1 | gaussian_hexagonal       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44125
    [    1.8s]   54/216 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43259
    [    1.8s]   55/216 | scene     fold1 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.35931
    [    1.8s]   56/216 | scene     fold1 | bubble_rectangular       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.50887
    [    1.9s]   57/216 | scene     fold1 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.37917
    [    1.9s]   58/216 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36057
    [    1.9s]   59/216 | scene     fold1 | bubble_rectangular       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.39753
    [    2.0s]   60/216 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36933
    [    2.0s]   61/216 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36837
    [    2.1s]   62/216 | scene     fold1 | bubble_rectangular       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36418
    [    2.1s]   63/216 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36887
    [    2.1s]   64/216 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.38254
    [    2.2s]   65/216 | scene     fold1 | bubble_hexagonal         | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.37142
    [    2.2s]   66/216 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.39544
    [    2.3s]   67/216 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37289
    [    2.3s]   68/216 | scene     fold1 | bubble_hexagonal         | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.39174
    [    2.4s]   69/216 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38050
    [    2.4s]   70/216 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36629
    [    2.4s]   71/216 | scene     fold1 | bubble_hexagonal         | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36360
    [    2.5s]   72/216 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38826
    [    2.5s]   73/216 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.57710
    [    2.5s]   74/216 | emotions  fold2 | gaussian_rectangular     | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.56896
    [    2.5s]   75/216 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.56956
    [    2.6s]   76/216 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59735
    [    2.6s]   77/216 | emotions  fold2 | gaussian_rectangular     | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57558
    [    2.6s]   78/216 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57661
    [    2.6s]   79/216 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.60071
    [    2.6s]   80/216 | emotions  fold2 | gaussian_rectangular     | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59581
    [    2.7s]   81/216 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59406
    [    2.7s]   82/216 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.56969
    [    2.7s]   83/216 | emotions  fold2 | gaussian_hexagonal       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.58492
    [    2.7s]   84/216 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.54954
    [    2.7s]   85/216 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55636
    [    2.7s]   86/216 | emotions  fold2 | gaussian_hexagonal       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56557
    [    2.8s]   87/216 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56582
    [    2.8s]   88/216 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.56068
    [    2.8s]   89/216 | emotions  fold2 | gaussian_hexagonal       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.58016
    [    2.8s]   90/216 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.55864
    [    2.8s]   91/216 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46311
    [    2.8s]   92/216 | emotions  fold2 | bubble_rectangular       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46238
    [    2.9s]   93/216 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46103
    [    2.9s]   94/216 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46601
    [    2.9s]   95/216 | emotions  fold2 | bubble_rectangular       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.50145
    [    2.9s]   96/216 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49363
    [    2.9s]   97/216 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49560
    [    2.9s]   98/216 | emotions  fold2 | bubble_rectangular       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.47860
    [    2.9s]   99/216 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.47886
    [    2.9s]  100/216 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.50593
    [    2.9s]  101/216 | emotions  fold2 | bubble_hexagonal         | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46123
    [    3.0s]  102/216 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45918
    [    3.0s]  103/216 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.47870
    [    3.0s]  104/216 | emotions  fold2 | bubble_hexagonal         | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49333
    [    3.0s]  105/216 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.47909
    [    3.0s]  106/216 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49356
    [    3.0s]  107/216 | emotions  fold2 | bubble_hexagonal         | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.50306
    [    3.0s]  108/216 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49509
    [    3.0s]  109/216 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45453
    [    3.1s]  110/216 | scene     fold2 | gaussian_rectangular     | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45008
    [    3.2s]  111/216 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45418
    [    3.2s]  112/216 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44935
    [    3.3s]  113/216 | scene     fold2 | gaussian_rectangular     | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46820
    [    3.4s]  114/216 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45619
    [    3.4s]  115/216 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45600
    [    3.5s]  116/216 | scene     fold2 | gaussian_rectangular     | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45625
    [    3.6s]  117/216 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45972
    [    3.6s]  118/216 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.44455
    [    3.7s]  119/216 | scene     fold2 | gaussian_hexagonal       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45266
    [    3.8s]  120/216 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.42966
    [    3.8s]  121/216 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43985
    [    3.9s]  122/216 | scene     fold2 | gaussian_hexagonal       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43688
    [    4.0s]  123/216 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43639
    [    4.0s]  124/216 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44803
    [    4.1s]  125/216 | scene     fold2 | gaussian_hexagonal       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45190
    [    4.2s]  126/216 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45288
    [    4.2s]  127/216 | scene     fold2 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.38245
    [    4.2s]  128/216 | scene     fold2 | bubble_rectangular       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.55560
    [    4.3s]  129/216 | scene     fold2 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.38064
    [    4.3s]  130/216 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.51553
    [    4.3s]  131/216 | scene     fold2 | bubble_rectangular       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37210
    [    4.4s]  132/216 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.39196
    [    4.4s]  133/216 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38306
    [    4.4s]  134/216 | scene     fold2 | bubble_rectangular       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38290
    [    4.5s]  135/216 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37814
    [    4.5s]  136/216 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.39591
    [    4.6s]  137/216 | scene     fold2 | bubble_hexagonal         | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.35840
    [    4.6s]  138/216 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.38148
    [    4.6s]  139/216 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38369
    [    4.6s]  140/216 | scene     fold2 | bubble_hexagonal         | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=1 QE=0.52969
    [    4.7s]  141/216 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38319
    [    4.7s]  142/216 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38375
    [    4.8s]  143/216 | scene     fold2 | bubble_hexagonal         | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37738
    [    4.8s]  144/216 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36618
    [    4.9s]  145/216 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.58185
    [    4.9s]  146/216 | emotions  fold3 | gaussian_rectangular     | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.59554
    [    4.9s]  147/216 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.59267
    [    4.9s]  148/216 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58547
    [    4.9s]  149/216 | emotions  fold3 | gaussian_rectangular     | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.61068
    [    4.9s]  150/216 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58048
    [    4.9s]  151/216 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59902
    [    5.0s]  152/216 | emotions  fold3 | gaussian_rectangular     | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.60285
    [    5.0s]  153/216 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.60067
    [    5.0s]  154/216 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.58913
    [    5.0s]  155/216 | emotions  fold3 | gaussian_hexagonal       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.54453
    [    5.0s]  156/216 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.56191
    [    5.1s]  157/216 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55844
    [    5.1s]  158/216 | emotions  fold3 | gaussian_hexagonal       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55158
    [    5.1s]  159/216 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56992
    [    5.1s]  160/216 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.55791
    [    5.1s]  161/216 | emotions  fold3 | gaussian_hexagonal       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.55574
    [    5.2s]  162/216 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.55531
    [    5.2s]  163/216 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.50412
    [    5.2s]  164/216 | emotions  fold3 | bubble_rectangular       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46559
    [    5.2s]  165/216 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.48276
    [    5.2s]  166/216 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46152
    [    5.2s]  167/216 | emotions  fold3 | bubble_rectangular       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46774
    [    5.2s]  168/216 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46762
    [    5.2s]  169/216 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.47931
    [    5.2s]  170/216 | emotions  fold3 | bubble_rectangular       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46013
    [    5.3s]  171/216 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46057
    [    5.3s]  172/216 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.48156
    [    5.3s]  173/216 | emotions  fold3 | bubble_hexagonal         | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45936
    [    5.3s]  174/216 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46822
    [    5.3s]  175/216 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48434
    [    5.3s]  176/216 | emotions  fold3 | bubble_hexagonal         | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46652
    [    5.3s]  177/216 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46207
    [    5.3s]  178/216 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45977
    [    5.3s]  179/216 | emotions  fold3 | bubble_hexagonal         | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48632
    [    5.4s]  180/216 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46606
    [    5.4s]  181/216 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45624
    [    5.4s]  182/216 | scene     fold3 | gaussian_rectangular     | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45416
    [    5.5s]  183/216 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45109
    [    5.6s]  184/216 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45893
    [    5.6s]  185/216 | scene     fold3 | gaussian_rectangular     | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45121
    [    5.7s]  186/216 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46463
    [    5.8s]  187/216 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45173
    [    5.8s]  188/216 | scene     fold3 | gaussian_rectangular     | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.47188
    [    5.9s]  189/216 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45848
    [    6.0s]  190/216 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45294
    [    6.0s]  191/216 | scene     fold3 | gaussian_hexagonal       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46309
    [    6.2s]  192/216 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.43571
    [    6.2s]  193/216 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46076
    [    6.3s]  194/216 | scene     fold3 | gaussian_hexagonal       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44778
    [    6.4s]  195/216 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44592
    [    6.4s]  196/216 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44284
    [    6.5s]  197/216 | scene     fold3 | gaussian_hexagonal       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43379
    [    6.6s]  198/216 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44464
    [    6.6s]  199/216 | scene     fold3 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.37024
    [    6.6s]  200/216 | scene     fold3 | bubble_rectangular       | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.36934
    [    6.7s]  201/216 | scene     fold3 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.37232
    [    6.7s]  202/216 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38467
    [    6.8s]  203/216 | scene     fold3 | bubble_rectangular       | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38441
    [    6.8s]  204/216 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38520
    [    6.9s]  205/216 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36839
    [    6.9s]  206/216 | scene     fold3 | bubble_rectangular       | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37751
    [    7.0s]  207/216 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36568
    [    7.0s]  208/216 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.39449
    [    7.0s]  209/216 | scene     fold3 | bubble_hexagonal         | rlen= 200 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.71993
    [    7.1s]  210/216 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.37219
    [    7.1s]  211/216 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36268
    [    7.1s]  212/216 | scene     fold3 | bubble_hexagonal         | rlen= 200 r=1.000000 a=(0.05,0.010) | empty=1 QE=0.55407
    [    7.2s]  213/216 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37227
    [    7.2s]  214/216 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37516
    [    7.2s]  215/216 | scene     fold3 | bubble_hexagonal         | rlen= 200 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36703
    [    7.3s]  216/216 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37015

``` r
results_df <- dplyr::bind_rows(results_list)
results_csv <- file.path(OUTPUT_PATH, sprintf("gridsearch_seed%d.csv", SEED))
write.csv(results_df, results_csv, row.names = FALSE)
cat(sprintf("\nResultados brutos salvos em: %s\n", results_csv))
```


    Resultados brutos salvos em: E:/git-tcc/tcc-final/outputs/gridsearch/gridsearch_seed1234.csv

## Seleção dos Melhores Parâmetros

``` r
# Estratégia:
# 1. Remove configurações com neurônios vazios ou falhas
# 2. Exige que a configuração tenha rodado nos 3 folds (estabilidade)
# 3. Calcula QE médio entre os folds
# 4. Seleciona o melhor por (dataset, vizinhança) — topologia também é fixada aqui
#    pois todos os 3 folds devem usar a mesma configuração (conforme o PDF)
best_params_mean <- results_df |>
    dplyr::filter(empty_neurons == 0, !is.na(qe_val)) |>
    dplyr::group_by(dataset, neighborhood, topology, rlen, radius, alpha_start, alpha_end) |>
    dplyr::summarise(
        mean_qe = mean(qe_val),
        sd_qe   = sd(qe_val),
        n_folds = dplyr::n(),
        .groups = "drop"
    ) |>
    dplyr::filter(n_folds == 3) |>
    dplyr::group_by(dataset, neighborhood) |>
    dplyr::slice_min(mean_qe, n = 1, with_ties = FALSE) |>
    dplyr::ungroup()

best_params_min <- results_df |>
    dplyr::filter(empty_neurons == 0, !is.na(qe_val)) |>
    dplyr::group_by(dataset, neighborhood, topology, rlen, radius, alpha_start, alpha_end) |>
    dplyr::summarise(
        min_qe  = min(qe_val),
        sd_qe   = sd(qe_val),
        n_folds = dplyr::n(),
        .groups = "drop"
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

    # A tibble: 4 × 10
      dataset  neighborhood topology     rlen radius alpha_start alpha_end mean_qe
      <chr>    <fct>        <chr>       <int>  <dbl>       <dbl>     <dbl>   <dbl>
    1 emotions bubble       rectangular   200    0.5        0.05      0.01   0.463
    2 emotions gaussian     hexagonal     200    1          0.05      0.01   0.556
    3 scene    bubble       hexagonal     200    1.5        0.05      0.01   0.369
    4 scene    gaussian     hexagonal     500    0.5        0.05      0.01   0.437
    # ℹ 2 more variables: sd_qe <dbl>, n_folds <int>

``` r
print(best_params_min)
```

    # A tibble: 4 × 10
      dataset  neighborhood topology   rlen radius alpha_start alpha_end min_qe
      <chr>    <fct>        <chr>     <int>  <dbl>       <dbl>     <dbl>  <dbl>
    1 emotions bubble       hexagonal   500    0.5        0.05      0.01  0.459
    2 emotions gaussian     hexagonal   500    1          0.05      0.01  0.544
    3 scene    bubble       hexagonal   200    0.5        0.05      0.01  0.358
    4 scene    gaussian     hexagonal   200    1          0.05      0.01  0.429
    # ℹ 2 more variables: sd_qe <dbl>, n_folds <int>

``` r
best_params_csv <- file.path(OUTPUT_PATH, "best_params.csv")
write.csv(best_params_mean, best_params_csv, row.names = FALSE)
cat(sprintf("\nMelhores parametros salvos em: %s\n", best_params_csv))
```


    Melhores parametros salvos em: E:/git-tcc/tcc-final/outputs/gridsearch/best_params.csv

``` r
best_params_csv_min <- file.path(OUTPUT_PATH, "best_params_min.csv")
write.csv(best_params_min, best_params_csv_min, row.names = FALSE)
cat(sprintf("Melhores parametros (min) salvos em: %s\n", best_params_csv_min))
```

    Melhores parametros (min) salvos em: E:/git-tcc/tcc-final/outputs/gridsearch/best_params_min.csv

## Sumário do Grid Search

``` r
cat("=== CONFIGURACOES COM NEURONIOS VAZIOS ===\n")
```

    === CONFIGURACOES COM NEURONIOS VAZIOS ===

``` r
results_df |>
    dplyr::filter(empty_neurons > 0) |>
    dplyr::count(dataset, neighborhood, topology, empty_neurons) |>
    print()
```

      dataset neighborhood  topology empty_neurons n
    1   scene       bubble hexagonal             1 2

``` r
cat("\n=== DISTRIBUICAO DE QE POR DATASET E VIZINHANCA ===\n")
```


    === DISTRIBUICAO DE QE POR DATASET E VIZINHANCA ===

``` r
results_df |>
    dplyr::filter(empty_neurons == 0, !is.na(qe_val)) |>
    dplyr::group_by(dataset, neighborhood) |>
    dplyr::summarise(
        n        = dplyr::n(),
        qe_min   = min(qe_val),
        qe_mediana = median(qe_val),
        qe_mean  = mean(qe_val),
        qe_max   = max(qe_val),
        .groups  = "drop"
    ) |>
    print()
```

    # A tibble: 4 × 7
      dataset  neighborhood     n qe_min qe_mediana qe_mean qe_max
      <chr>    <fct>        <int>  <dbl>      <dbl>   <dbl>  <dbl>
    1 emotions bubble          54  0.459      0.479   0.479  0.516
    2 emotions gaussian        54  0.544      0.577   0.577  0.611
    3 scene    bubble          52  0.358      0.378   0.392  0.720
    4 scene    gaussian        54  0.429      0.451   0.450  0.472

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

# Script 1 — Grid Search


## Configuração

``` r
library(tidyverse)
library(kohonen)

BASE_PATH <- "E:/git-tcc/tcc-final/data"
OUTPUT_PATH <- "E:/git-tcc/tcc-final/outputs/gridsearch"
SEED <- 1234L
# Conforme o PDF: apenas emotions e scene
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

## Definição das Grades SOM (2×2 apenas)

``` r
# Conforme o PDF: apenas grade 2x2, 2 vizinhancas x 2 topologias = 4 grids
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
    rlen = c(100L, 500L, 1000L), # default: 100L
    radius = c(0.5, 1.0, 1.5, 2.0), # default: 1.0?
    alpha_start = c(0.05), # default: 0.05
    alpha_end = c(0.01), # default: 0.01
    grid_name = names(grids),
    dataset = TARGET_DATASETS,
    fold = 1:3,
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

    Total de combinacoes: 288
      (4 grids x 2 datasets x 3 folds x 12 combinacoes de params)

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

    [    0.1s]    1/288 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.59277
    [    0.1s]    2/288 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.59399
    [    0.2s]    3/288 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.58610
    [    0.2s]    4/288 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59381
    [    0.2s]    5/288 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58205
    [    0.2s]    6/288 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59300
    [    0.2s]    7/288 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59409
    [    0.3s]    8/288 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59234
    [    0.3s]    9/288 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59340
    [    0.3s]   10/288 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.60478
    [    0.4s]   11/288 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.59725
    [    0.4s]   12/288 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.57448
    [    0.5s]   13/288 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.54484
    [    0.5s]   14/288 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.57618
    [    0.5s]   15/288 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.54917
    [    0.5s]   16/288 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57652
    [    0.6s]   17/288 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55449
    [    0.6s]   18/288 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56627
    [    0.6s]   19/288 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54133
    [    0.7s]   20/288 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.55913
    [    0.7s]   21/288 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.56358
    [    0.7s]   22/288 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.57199
    [    0.8s]   23/288 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.56027
    [    0.8s]   24/288 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.57499
    [    0.8s]   25/288 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.48359
    [    0.8s]   26/288 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.48522
    [    0.8s]   27/288 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.48104
    [    0.8s]   28/288 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49533
    [    0.9s]   29/288 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45952
    [    0.9s]   30/288 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45970
    [    0.9s]   31/288 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48701
    [    0.9s]   32/288 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49570
    [    0.9s]   33/288 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46768
    [    0.9s]   34/288 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.50981
    [    1.0s]   35/288 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45920
    [    1.0s]   36/288 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.51036
    [    1.0s]   37/288 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.51063
    [    1.0s]   38/288 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46791
    [    1.0s]   39/288 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46658
    [    1.0s]   40/288 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46304
    [    1.1s]   41/288 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46054
    [    1.1s]   42/288 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46681
    [    1.1s]   43/288 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49791
    [    1.1s]   44/288 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.51004
    [    1.1s]   45/288 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.49635
    [    1.1s]   46/288 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.49848
    [    1.2s]   47/288 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.46024
    [    1.2s]   48/288 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.50130
    [    1.2s]   49/288 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.44774
    [    1.3s]   50/288 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45193
    [    1.5s]   51/288 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.44890
    [    1.5s]   52/288 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45773
    [    1.6s]   53/288 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44100
    [    1.8s]   54/288 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45271
    [    1.9s]   55/288 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44968
    [    2.0s]   56/288 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45450
    [    2.2s]   57/288 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45628
    [    2.2s]   58/288 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45472
    [    2.3s]   59/288 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44286
    [    2.5s]   60/288 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44149
    [    2.6s]   61/288 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.44420
    [    2.7s]   62/288 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.43912
    [    2.9s]   63/288 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.43298
    [    2.9s]   64/288 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44753
    [    3.0s]   65/288 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.42868
    [    3.2s]   66/288 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44166
    [    3.2s]   67/288 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44043
    [    3.3s]   68/288 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44122
    [    3.5s]   69/288 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43494
    [    3.5s]   70/288 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44711
    [    3.6s]   71/288 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.42995
    [    3.8s]   72/288 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.43560
    [    3.9s]   73/288 | scene     fold1 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.36911
    [    3.9s]   74/288 | scene     fold1 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.38181
    [    4.0s]   75/288 | scene     fold1 | bubble_rectangular       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.36993
    [    4.0s]   76/288 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38168
    [    4.1s]   77/288 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37901
    [    4.2s]   78/288 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.39439
    [    4.2s]   79/288 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36979
    [    4.3s]   80/288 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38034
    [    4.4s]   81/288 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38309
    [    4.5s]   82/288 | scene     fold1 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36707
    [    4.5s]   83/288 | scene     fold1 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36458
    [    4.7s]   84/288 | scene     fold1 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38046
    [    4.7s]   85/288 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.39113
    [    4.7s]   86/288 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.35979
    [    4.8s]   87/288 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.36032
    [    4.8s]   88/288 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37857
    [    4.9s]   89/288 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37482
    [    5.0s]   90/288 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36037
    [    5.0s]   91/288 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36657
    [    5.1s]   92/288 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37087
    [    5.2s]   93/288 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38163
    [    5.2s]   94/288 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.39150
    [    5.3s]   95/288 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38284
    [    5.4s]   96/288 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36239
    [    5.4s]   97/288 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.58628
    [    5.4s]   98/288 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.58988
    [    5.5s]   99/288 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.59109
    [    5.5s]  100/288 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57201
    [    5.5s]  101/288 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57770
    [    5.6s]  102/288 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58995
    [    5.6s]  103/288 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59003
    [    5.6s]  104/288 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.58819
    [    5.7s]  105/288 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59256
    [    5.7s]  106/288 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.60682
    [    5.7s]  107/288 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.59458
    [    5.8s]  108/288 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.59117
    [    5.8s]  109/288 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.54890
    [    5.8s]  110/288 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.53890
    [    5.8s]  111/288 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.53973
    [    5.8s]  112/288 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.54571
    [    5.9s]  113/288 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55540
    [    5.9s]  114/288 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.54535
    [    5.9s]  115/288 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.57187
    [    5.9s]  116/288 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54905
    [    6.0s]  117/288 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54989
    [    6.0s]  118/288 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.56161
    [    6.0s]  119/288 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.56659
    [    6.1s]  120/288 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.55017
    [    6.1s]  121/288 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46350
    [    6.1s]  122/288 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.47629
    [    6.2s]  123/288 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45805
    [    6.2s]  124/288 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46143
    [    6.2s]  125/288 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46262
    [    6.2s]  126/288 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.50161
    [    6.2s]  127/288 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.47952
    [    6.2s]  128/288 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.47889
    [    6.2s]  129/288 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46137
    [    6.2s]  130/288 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.47725
    [    6.3s]  131/288 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.47973
    [    6.3s]  132/288 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.48073
    [    6.3s]  133/288 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46367
    [    6.3s]  134/288 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45868
    [    6.3s]  135/288 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46314
    [    6.3s]  136/288 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49350
    [    6.4s]  137/288 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46118
    [    6.4s]  138/288 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.47678
    [    6.4s]  139/288 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46346
    [    6.4s]  140/288 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45851
    [    6.4s]  141/288 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.47912
    [    6.4s]  142/288 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.50091
    [    6.4s]  143/288 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.49568
    [    6.5s]  144/288 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.47928
    [    6.5s]  145/288 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45850
    [    6.6s]  146/288 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46829
    [    6.8s]  147/288 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45809
    [    6.8s]  148/288 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44461
    [    6.9s]  149/288 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45130
    [    7.1s]  150/288 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44897
    [    7.1s]  151/288 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45644
    [    7.3s]  152/288 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44458
    [    7.4s]  153/288 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45306
    [    7.5s]  154/288 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45618
    [    7.6s]  155/288 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45225
    [    7.8s]  156/288 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44450
    [    7.9s]  157/288 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.44201
    [    8.0s]  158/288 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.42422
    [    8.1s]  159/288 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.42829
    [    8.2s]  160/288 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46257
    [    8.3s]  161/288 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43702
    [    8.5s]  162/288 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43320
    [    8.5s]  163/288 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43696
    [    8.6s]  164/288 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43661
    [    8.8s]  165/288 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43695
    [    8.8s]  166/288 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.43312
    [    8.9s]  167/288 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44477
    [    9.2s]  168/288 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44835
    [    9.2s]  169/288 | scene     fold2 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.37985
    [    9.2s]  170/288 | scene     fold2 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.51264
    [    9.3s]  171/288 | scene     fold2 | bubble_rectangular       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.36318
    [    9.3s]  172/288 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38024
    [    9.4s]  173/288 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56128
    [    9.5s]  174/288 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36913
    [    9.5s]  175/288 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37135
    [    9.6s]  176/288 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37683
    [    9.7s]  177/288 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38159
    [    9.7s]  178/288 | scene     fold2 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38193
    [    9.8s]  179/288 | scene     fold2 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38193
    [    9.9s]  180/288 | scene     fold2 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36120
    [    9.9s]  181/288 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.36988
    [   10.0s]  182/288 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.38550
    [   10.1s]  183/288 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.37439
    [   10.1s]  184/288 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37003
    [   10.1s]  185/288 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.39196
    [   10.2s]  186/288 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37198
    [   10.3s]  187/288 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37122
    [   10.3s]  188/288 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36900
    [   10.5s]  189/288 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38357
    [   10.5s]  190/288 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38437
    [   10.5s]  191/288 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36631
    [   10.7s]  192/288 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.35878
    [   10.7s]  193/288 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.58135
    [   10.7s]  194/288 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.57966
    [   10.8s]  195/288 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.57559
    [   10.8s]  196/288 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59302
    [   10.8s]  197/288 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57608
    [   10.8s]  198/288 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59610
    [   10.9s]  199/288 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.60929
    [   10.9s]  200/288 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59654
    [   10.9s]  201/288 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.59795
    [   10.9s]  202/288 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.60378
    [   11.0s]  203/288 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.60344
    [   11.0s]  204/288 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.61035
    [   11.0s]  205/288 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.57474
    [   11.1s]  206/288 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.55925
    [   11.1s]  207/288 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.55396
    [   11.1s]  208/288 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56434
    [   11.1s]  209/288 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55115
    [   11.2s]  210/288 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.54109
    [   11.2s]  211/288 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.53823
    [   11.2s]  212/288 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.55848
    [   11.3s]  213/288 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.54322
    [   11.3s]  214/288 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.57549
    [   11.3s]  215/288 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.57748
    [   11.4s]  216/288 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.56411
    [   11.4s]  217/288 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46653
    [   11.4s]  218/288 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45881
    [   11.4s]  219/288 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.46796
    [   11.4s]  220/288 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.50128
    [   11.5s]  221/288 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48175
    [   11.5s]  222/288 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46580
    [   11.5s]  223/288 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48525
    [   11.5s]  224/288 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45895
    [   11.5s]  225/288 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48353
    [   11.5s]  226/288 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.48509
    [   11.6s]  227/288 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45879
    [   11.6s]  228/288 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.50801
    [   11.6s]  229/288 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45850
    [   11.6s]  230/288 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.49875
    [   11.6s]  231/288 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45786
    [   11.6s]  232/288 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48363
    [   11.6s]  233/288 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46532
    [   11.7s]  234/288 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46294
    [   11.7s]  235/288 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46117
    [   11.7s]  236/288 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.48206
    [   11.7s]  237/288 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.46692
    [   11.8s]  238/288 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.49912
    [   11.8s]  239/288 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45910
    [   11.8s]  240/288 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.50052
    [   11.9s]  241/288 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45350
    [   11.9s]  242/288 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.44877
    [   12.2s]  243/288 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.45373
    [   12.2s]  244/288 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44149
    [   12.3s]  245/288 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46234
    [   12.5s]  246/288 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45347
    [   12.5s]  247/288 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44588
    [   12.6s]  248/288 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.45261
    [   12.8s]  249/288 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44026
    [   12.9s]  250/288 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44721
    [   13.0s]  251/288 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45102
    [   13.2s]  252/288 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.43958
    [   13.2s]  253/288 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.43232
    [   13.3s]  254/288 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.42770
    [   13.5s]  255/288 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.43410
    [   13.5s]  256/288 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43359
    [   13.6s]  257/288 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44698
    [   13.8s]  258/288 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45146
    [   13.8s]  259/288 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.43602
    [   13.9s]  260/288 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44618
    [   14.1s]  261/288 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.44224
    [   14.2s]  262/288 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44163
    [   14.3s]  263/288 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.42923
    [   14.5s]  264/288 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45943
    [   14.5s]  265/288 | scene     fold3 | bubble_rectangular       | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.39069
    [   14.5s]  266/288 | scene     fold3 | bubble_rectangular       | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.35982
    [   14.6s]  267/288 | scene     fold3 | bubble_rectangular       | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.38997
    [   14.6s]  268/288 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38118
    [   14.7s]  269/288 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36946
    [   14.8s]  270/288 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38044
    [   14.8s]  271/288 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37730
    [   14.9s]  272/288 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37023
    [   15.0s]  273/288 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37125
    [   15.1s]  274/288 | scene     fold3 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36645
    [   15.1s]  275/288 | scene     fold3 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38086
    [   15.2s]  276/288 | scene     fold3 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36710
    [   15.3s]  277/288 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.39404
    [   15.3s]  278/288 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.38162
    [   15.4s]  279/288 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=0.500000 a=(0.05,0.010) | empty=0 QE=0.38484
    [   15.4s]  280/288 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.50849
    [   15.5s]  281/288 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38114
    [   15.6s]  282/288 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38746
    [   15.6s]  283/288 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.36991
    [   15.7s]  284/288 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.37461
    [   15.8s]  285/288 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.500000 a=(0.05,0.010) | empty=0 QE=0.38406
    [   15.8s]  286/288 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38500
    [   15.9s]  287/288 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38385
    [   16.0s]  288/288 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38053

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
    dplyr::filter(empty_neurons == 0, !is.na(qe_val)) |>
    dplyr::group_by(dataset, neighborhood) |>
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

    [1] dataset       neighborhood  topology      empty_neurons n            
    <0 rows> (or 0-length row.names)

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

    # A tibble: 4 × 7
      dataset  neighborhood     n qe_min qe_mediana qe_mean qe_max
      <chr>    <fct>        <int>  <dbl>      <dbl>   <dbl>  <dbl>
    1 emotions bubble          72  0.458      0.477   0.478  0.511
    2 emotions gaussian        72  0.538      0.576   0.574  0.610
    3 scene    bubble          72  0.359      0.379   0.383  0.561
    4 scene    gaussian        72  0.424      0.445   0.445  0.468

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
    dplyr::reframe(
        mean_qe = mean(qe_val),
        sd_qe   = sd(qe_val),
        n_folds = dplyr::n()
    ) |>
    dplyr::filter(n_folds == 3) |>
    dplyr::group_by(dataset, neighborhood) |>
    dplyr::slice_min(mean_qe, n = 1, with_ties = FALSE) |>
    dplyr::ungroup()

best_params_min <- results_df |>
    dplyr::filter(empty_neurons == 0, !is.na(qe_val)) |>
    dplyr::group_by(dataset, neighborhood, topology, rlen, radius, alpha_start, alpha_end) |>
    dplyr::reframe(
        min_qe  = min(qe_val),
        mean_qe = mean(qe_val),
        sd_qe   = sd(qe_val),
        n_folds = dplyr::n()
    ) |>
    dplyr::filter(n_folds == 3) |>
    dplyr::group_by(dataset, neighborhood) |>
    dplyr::slice_min(min_qe, n = 1, with_ties = FALSE) |>
    dplyr::ungroup()

best_params_search <- results_df |>
    dplyr::filter(empty_neurons == 0, !is.na(qe_val)) |>
    dplyr::group_by(dataset, neighborhood, topology, rlen, radius, alpha_start, alpha_end) |>
    dplyr::reframe(
        qe_val  = qe_val,
        min_qe  = min(qe_val),
        mean_qe = mean(qe_val),
        sd_qe   = sd(qe_val),
        n_folds = dplyr::n()
    ) |>
    dplyr::filter(n_folds == 3) |>
    dplyr::group_by(dataset, neighborhood) |>
    dplyr::slice_min(min_qe, n = 9, with_ties = FALSE) |>
    dplyr::ungroup()

cat("=== MELHORES PARAMETROS POR DATASET E VIZINHANCA ===\n")
```

    === MELHORES PARAMETROS POR DATASET E VIZINHANCA ===

``` r
print(best_params_mean)
```

    # A tibble: 4 × 10
      dataset  neighborhood topology   rlen radius alpha_start alpha_end mean_qe
      <chr>    <fct>        <chr>     <int>  <dbl>       <dbl>     <dbl>   <dbl>
    1 emotions bubble       hexagonal   500    1          0.05      0.01   0.462
    2 emotions gaussian     hexagonal  1000    0.5        0.05      0.01   0.548
    3 scene    bubble       hexagonal  1000    2          0.05      0.01   0.367
    4 scene    gaussian     hexagonal   500    0.5        0.05      0.01   0.430
    # ℹ 2 more variables: sd_qe <dbl>, n_folds <int>

``` r
print(best_params_min)
```

    # A tibble: 4 × 11
      dataset  neighborhood topology   rlen radius alpha_start alpha_end min_qe
      <chr>    <fct>        <chr>     <int>  <dbl>       <dbl>     <dbl>  <dbl>
    1 emotions bubble       hexagonal  1000    0.5        0.05      0.01  0.458
    2 emotions gaussian     hexagonal   100    1.5        0.05      0.01  0.538
    3 scene    bubble       hexagonal  1000    2          0.05      0.01  0.359
    4 scene    gaussian     hexagonal   500    0.5        0.05      0.01  0.424
    # ℹ 3 more variables: mean_qe <dbl>, sd_qe <dbl>, n_folds <int>

``` r
print(best_params_search)
```

    # A tibble: 36 × 12
       dataset  neighborhood topology     rlen radius alpha_start alpha_end qe_val
       <chr>    <fct>        <chr>       <int>  <dbl>       <dbl>     <dbl>  <dbl>
     1 emotions bubble       hexagonal    1000    0.5        0.05      0.01  0.467
     2 emotions bubble       hexagonal    1000    0.5        0.05      0.01  0.463
     3 emotions bubble       hexagonal    1000    0.5        0.05      0.01  0.458
     4 emotions bubble       rectangular  1000    0.5        0.05      0.01  0.481
     5 emotions bubble       rectangular  1000    0.5        0.05      0.01  0.458
     6 emotions bubble       rectangular  1000    0.5        0.05      0.01  0.468
     7 emotions bubble       hexagonal     100    0.5        0.05      0.01  0.511
     8 emotions bubble       hexagonal     100    0.5        0.05      0.01  0.464
     9 emotions bubble       hexagonal     100    0.5        0.05      0.01  0.459
    10 emotions gaussian     hexagonal     100    1.5        0.05      0.01  0.541
    # ℹ 26 more rows
    # ℹ 4 more variables: min_qe <dbl>, mean_qe <dbl>, sd_qe <dbl>, n_folds <int>

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
best_params_csv_search <- file.path(OUTPUT_PATH, "best_params_search.csv")
write.csv(best_params_search, best_params_csv_search, row.names = FALSE)
cat(sprintf("Melhores parametros (busca) salvos em: %s\n", best_params_csv_search))
```

    Melhores parametros (busca) salvos em: E:/git-tcc/tcc-final/outputs/gridsearch/best_params_search.csv

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

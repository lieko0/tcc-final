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
    rlen = c(100L, 500L, 1000L), # default: 100L
    radius = c(1.0, 2.0), # default: 1.0?
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

    Total de combinacoes: 144
      (4 grids x 2 datasets x 3 folds x 6 combinacoes de params)

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

    [    0.1s]    1/144 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59277
    [    0.1s]    2/144 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59399
    [    0.2s]    3/144 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58610
    [    0.2s]    4/144 | emotions  fold1 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.60174
    [    0.2s]    5/144 | emotions  fold1 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.59162
    [    0.3s]    6/144 | emotions  fold1 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.60425
    [    0.3s]    7/144 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56056
    [    0.3s]    8/144 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55244
    [    0.4s]    9/144 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.56879
    [    0.4s]   10/144 | emotions  fold1 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.54211
    [    0.4s]   11/144 | emotions  fold1 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.57622
    [    0.5s]   12/144 | emotions  fold1 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.55200
    [    0.5s]   13/144 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48185
    [    0.5s]   14/144 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46703
    [    0.5s]   15/144 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48585
    [    0.5s]   16/144 | emotions  fold1 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.46328
    [    0.5s]   17/144 | emotions  fold1 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.48185
    [    0.6s]   18/144 | emotions  fold1 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.46694
    [    0.6s]   19/144 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48102
    [    0.6s]   20/144 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49527
    [    0.6s]   21/144 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46770
    [    0.6s]   22/144 | emotions  fold1 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.48605
    [    0.6s]   23/144 | emotions  fold1 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.50546
    [    0.7s]   24/144 | emotions  fold1 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.49681
    [    0.7s]   25/144 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45565
    [    0.8s]   26/144 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46205
    [    1.0s]   27/144 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45002
    [    1.0s]   28/144 | scene     fold1 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.47305
    [    1.1s]   29/144 | scene     fold1 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44624
    [    1.4s]   30/144 | scene     fold1 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44670
    [    1.4s]   31/144 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43942
    [    1.5s]   32/144 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44560
    [    1.7s]   33/144 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44141
    [    1.7s]   34/144 | scene     fold1 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44001
    [    1.8s]   35/144 | scene     fold1 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.43821
    [    2.1s]   36/144 | scene     fold1 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.43330
    [    2.1s]   37/144 | scene     fold1 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37970
    [    2.1s]   38/144 | scene     fold1 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38170
    [    2.3s]   39/144 | scene     fold1 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.35867
    [    2.3s]   40/144 | scene     fold1 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38335
    [    2.4s]   41/144 | scene     fold1 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.39423
    [    2.5s]   42/144 | scene     fold1 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36871
    [    2.5s]   43/144 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38071
    [    2.6s]   44/144 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.35881
    [    2.7s]   45/144 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37399
    [    2.7s]   46/144 | scene     fold1 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36641
    [    2.8s]   47/144 | scene     fold1 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36597
    [    2.9s]   48/144 | scene     fold1 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38721
    [    2.9s]   49/144 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57635
    [    2.9s]   50/144 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57828
    [    3.0s]   51/144 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57208
    [    3.0s]   52/144 | emotions  fold2 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.60650
    [    3.0s]   53/144 | emotions  fold2 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.60276
    [    3.1s]   54/144 | emotions  fold2 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.59203
    [    3.1s]   55/144 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55041
    [    3.1s]   56/144 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.54652
    [    3.2s]   57/144 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59184
    [    3.2s]   58/144 | emotions  fold2 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.54702
    [    3.2s]   59/144 | emotions  fold2 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.54701
    [    3.3s]   60/144 | emotions  fold2 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.55129
    [    3.3s]   61/144 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46072
    [    3.3s]   62/144 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45931
    [    3.3s]   63/144 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49407
    [    3.3s]   64/144 | emotions  fold2 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45870
    [    3.3s]   65/144 | emotions  fold2 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.48633
    [    3.4s]   66/144 | emotions  fold2 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.50185
    [    3.4s]   67/144 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.47926
    [    3.4s]   68/144 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.47851
    [    3.4s]   69/144 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45959
    [    3.4s]   70/144 | emotions  fold2 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.46364
    [    3.4s]   71/144 | emotions  fold2 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.50269
    [    3.5s]   72/144 | emotions  fold2 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.49496
    [    3.5s]   73/144 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.44158
    [    3.6s]   74/144 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45217
    [    3.8s]   75/144 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45543
    [    3.8s]   76/144 | scene     fold2 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45303
    [    4.0s]   77/144 | scene     fold2 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45043
    [    4.2s]   78/144 | scene     fold2 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45699
    [    4.2s]   79/144 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43112
    [    4.3s]   80/144 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43489
    [    4.5s]   81/144 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43192
    [    4.5s]   82/144 | scene     fold2 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.42868
    [    4.7s]   83/144 | scene     fold2 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45320
    [    4.9s]   84/144 | scene     fold2 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44583
    [    4.9s]   85/144 | scene     fold2 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36199
    [    5.0s]   86/144 | scene     fold2 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=1 QE=0.55655
    [    5.1s]   87/144 | scene     fold2 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36568
    [    5.1s]   88/144 | scene     fold2 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36839
    [    5.1s]   89/144 | scene     fold2 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38987
    [    5.3s]   90/144 | scene     fold2 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.37278
    [    5.3s]   91/144 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38251
    [    5.4s]   92/144 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.38154
    [    5.5s]   93/144 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37285
    [    5.5s]   94/144 | scene     fold2 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38333
    [    5.6s]   95/144 | scene     fold2 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36730
    [    5.7s]   96/144 | scene     fold2 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36683
    [    5.7s]   97/144 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.59142
    [    5.7s]   98/144 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.57309
    [    5.8s]   99/144 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.58665
    [    5.8s]  100/144 | emotions  fold3 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.59129
    [    5.8s]  101/144 | emotions  fold3 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.60534
    [    5.9s]  102/144 | emotions  fold3 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.58929
    [    5.9s]  103/144 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.54534
    [    5.9s]  104/144 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55679
    [    5.9s]  105/144 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.55790
    [    6.0s]  106/144 | emotions  fold3 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.55660
    [    6.0s]  107/144 | emotions  fold3 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.57409
    [    6.0s]  108/144 | emotions  fold3 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.56230
    [    6.1s]  109/144 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46505
    [    6.1s]  110/144 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.49750
    [    6.1s]  111/144 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46510
    [    6.1s]  112/144 | emotions  fold3 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.48313
    [    6.1s]  113/144 | emotions  fold3 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.48368
    [    6.1s]  114/144 | emotions  fold3 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.50689
    [    6.1s]  115/144 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.46047
    [    6.2s]  116/144 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.50200
    [    6.2s]  117/144 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.48431
    [    6.2s]  118/144 | emotions  fold3 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.50882
    [    6.2s]  119/144 | emotions  fold3 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.49934
    [    6.2s]  120/144 | emotions  fold3 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.48361
    [    6.3s]  121/144 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45729
    [    6.4s]  122/144 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43997
    [    6.6s]  123/144 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.45997
    [    6.6s]  124/144 | scene     fold3 | gaussian_rectangular     | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.46046
    [    6.8s]  125/144 | scene     fold3 | gaussian_rectangular     | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.46105
    [    7.0s]  126/144 | scene     fold3 | gaussian_rectangular     | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.45670
    [    7.0s]  127/144 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43619
    [    7.1s]  128/144 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.42616
    [    7.3s]  129/144 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.43932
    [    7.4s]  130/144 | scene     fold3 | gaussian_hexagonal       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44373
    [    7.5s]  131/144 | scene     fold3 | gaussian_hexagonal       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.43787
    [    7.7s]  132/144 | scene     fold3 | gaussian_hexagonal       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.44303
    [    7.7s]  133/144 | scene     fold3 | bubble_rectangular       | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37914
    [    7.8s]  134/144 | scene     fold3 | bubble_rectangular       | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.35892
    [    7.9s]  135/144 | scene     fold3 | bubble_rectangular       | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37170
    [    7.9s]  136/144 | scene     fold3 | bubble_rectangular       | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.37271
    [    8.0s]  137/144 | scene     fold3 | bubble_rectangular       | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.36374
    [    8.1s]  138/144 | scene     fold3 | bubble_rectangular       | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38120
    [    8.1s]  139/144 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.37104
    [    8.2s]  140/144 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.50867
    [    8.3s]  141/144 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=1.000000 a=(0.05,0.010) | empty=0 QE=0.36797
    [    8.3s]  142/144 | scene     fold3 | bubble_hexagonal         | rlen= 100 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38216
    [    8.4s]  143/144 | scene     fold3 | bubble_hexagonal         | rlen= 500 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.38275
    [    8.5s]  144/144 | scene     fold3 | bubble_hexagonal         | rlen=1000 r=2.000000 a=(0.05,0.010) | empty=0 QE=0.37025

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

      dataset neighborhood    topology empty_neurons n
    1   scene       bubble rectangular             1 1

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
    1 emotions bubble          36  0.459      0.483   0.482  0.509
    2 emotions gaussian        36  0.542      0.575   0.574  0.606
    3 scene    bubble          35  0.359      0.373   0.378  0.509
    4 scene    gaussian        36  0.426      0.446   0.446  0.473

## Seleção dos Melhores Parâmetros

``` r
# Estratégia:
# 1. Remove configurações com neurônios vazios ou falhas
# 2. Exige que a configuração tenha rodado nos 3 folds (estabilidade)
# 3. Calcula QE médio entre os folds
# 4. Seleciona o melhor por (dataset, vizinhança) — topologia também é fixada aqui
#    pois todos os 3 folds devem usar a mesma configuração
best_params_mean <- results_df |>
    dplyr::filter(!is.na(qe_val)) |>
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
    dplyr::filter(!is.na(qe_val)) |>
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
    dplyr::filter(!is.na(qe_val)) |>
    dplyr::group_by(dataset, neighborhood, topology, rlen, radius, alpha_start, alpha_end) |>
    dplyr::reframe(
        qe_val  = qe_val,
        n_folds = dplyr::n(),
        min_qe  = min(qe_val),
        mean_qe = mean(qe_val),
        sd_qe   = sd(qe_val),
        fold_num = fold,
        empty_count = sum(empty_neurons > 0)
    ) |>
    dplyr::filter(n_folds == 3) |>
    dplyr::group_by(dataset, neighborhood) |>
    dplyr::slice_min(min_qe, n = 9, with_ties = FALSE) |>
    dplyr::ungroup()

best_params_search2 <- results_df |>
    dplyr::filter(!is.na(qe_val)) |>
    dplyr::group_by(dataset, neighborhood, topology, rlen, radius, alpha_start, alpha_end) |>
    dplyr::reframe(
        qe_val  = qe_val,
        min_qe  = min(qe_val),
        mean_qe = mean(qe_val),
        sd_qe   = sd(qe_val),
        fold_num = fold,
        n_folds = dplyr::n(),
        empty_count = sum(empty_neurons > 0)
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
      dataset  neighborhood topology     rlen radius alpha_start alpha_end mean_qe
      <chr>    <fct>        <chr>       <int>  <dbl>       <dbl>     <dbl>   <dbl>
    1 emotions bubble       rectangular   100      2        0.05      0.01   0.468
    2 emotions gaussian     hexagonal     100      2        0.05      0.01   0.549
    3 scene    bubble       rectangular  1000      1        0.05      0.01   0.365
    4 scene    gaussian     hexagonal     500      1        0.05      0.01   0.436
    # ℹ 2 more variables: sd_qe <dbl>, n_folds <int>

``` r
print(best_params_min)
```

    # A tibble: 4 × 11
      dataset  neighborhood topology     rlen radius alpha_start alpha_end min_qe
      <chr>    <fct>        <chr>       <int>  <dbl>       <dbl>     <dbl>  <dbl>
    1 emotions bubble       rectangular   100      2        0.05      0.01  0.459
    2 emotions gaussian     hexagonal     100      2        0.05      0.01  0.542
    3 scene    bubble       rectangular  1000      1        0.05      0.01  0.359
    4 scene    gaussian     hexagonal     500      1        0.05      0.01  0.426
    # ℹ 3 more variables: mean_qe <dbl>, sd_qe <dbl>, n_folds <int>

``` r
print(best_params_search)
```

    # A tibble: 36 × 14
       dataset  neighborhood topology     rlen radius alpha_start alpha_end qe_val
       <chr>    <fct>        <chr>       <int>  <dbl>       <dbl>     <dbl>  <dbl>
     1 emotions bubble       rectangular   100      2        0.05      0.01  0.463
     2 emotions bubble       rectangular   100      2        0.05      0.01  0.459
     3 emotions bubble       rectangular   100      2        0.05      0.01  0.483
     4 emotions bubble       rectangular   500      1        0.05      0.01  0.467
     5 emotions bubble       rectangular   500      1        0.05      0.01  0.459
     6 emotions bubble       rectangular   500      1        0.05      0.01  0.497
     7 emotions bubble       hexagonal    1000      1        0.05      0.01  0.468
     8 emotions bubble       hexagonal    1000      1        0.05      0.01  0.460
     9 emotions bubble       hexagonal    1000      1        0.05      0.01  0.484
    10 emotions gaussian     hexagonal     100      2        0.05      0.01  0.542
    # ℹ 26 more rows
    # ℹ 6 more variables: n_folds <int>, min_qe <dbl>, mean_qe <dbl>, sd_qe <dbl>,
    #   fold_num <int>, empty_count <int>

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
best_params_csv_search2 <- file.path(OUTPUT_PATH, "best_params_search2.csv")
write.csv(best_params_search2, best_params_csv_search2, row.names = FALSE)
cat(sprintf("Melhores parametros (busca 2) salvos em: %s\n", best_params_csv_search2))
```

    Melhores parametros (busca 2) salvos em: E:/git-tcc/tcc-final/outputs/gridsearch/best_params_search2.csv

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

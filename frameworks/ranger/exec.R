library(mlr)
library(mlrCPO)
library(farff)


run <- function(train_file, test_file, output_predictions_file, cores=1, meta_results_file=NULL) {
  train <- readARFF(train_file)
  test <- readARFF(test_file)
  colnames(train) <- make.names(colnames(train))
  colnames(test) <- make.names(colnames(test))
  target <- colnames(train)[ncol(train)]
  train <- makeClassifTask(data = train, target = target)

  lrn <- cpoDropConstants() %>>%
        cpoImputeAll(classes = list(numeric = imputeMax(2),
                                    integer = imputeMax(2),
                                    factor = imputeConstant("__MISS__"))
                    ) %>>%
        makeLearner("classif.ranger", num.threads = cores, predict.type = "prob")

  mod <- NULL
  preds <- NULL
  training <- function() mod <<- train(lrn, train)
  prediction <- function() preds <<- predict(mod, newdata = test)$data
  train_duration <- system.time(training())[['elapsed']]
  predict_duration <- system.time(prediction())[['elapsed']]

  preds <- preds[c(2:ncol(preds), 1)]
  names(preds)[names(preds) == "response"] <- "predictions"
  names(preds) <- sub("^prob.", "", names(preds))
  # FIXME: label encoding for predictions and truth?

  write.csv(preds, file = output_predictions_file, row.names = FALSE)

  meta_results <- data.frame(key=c("training_duration", "predict_duration"),
                             value=c(train_duration, predict_duration))
  write.csv(meta_results, file = meta_results_file, row.names = FALSE)
}

# args = commandArgs(trailingOnly=TRUE)
# run(args[1], args[2], args[3], as.integer(args[4]))

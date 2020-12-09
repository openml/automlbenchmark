library(mlr)
library(autoxgboost)
library(farff)

run <- function(train_file, test_file, target.index, type, output_predictions_file, cores, time.budget, meta_results_file) {
  train <- farff::readARFF(train_file)
  colnames(train) <- make.names(colnames(train))
  target <- colnames(train)[target.index]

  if (type == "classification") {
    train <- makeClassifTask(data = train, target = target)
    lrn <- makeLearner("classif.autoxgboost", time.budget = time.budget, nthread = cores, predict.type = "prob")
  } else if (type == "regression") {
    train <- makeRegrTask(data = train, target = target)
    lrn <- makeLearner("regr.autoxgboost", time.budget = time.budget, nthread = cores)
  } else {
    stop("Task type not supported!")
  }

  mod <- NULL
  preds <- NULL
  training <- function() mod <<- train(lrn, train)
  prediction <- function() preds <<- predict(mod, newdata = test)$data

  train_duration <- system.time(training())[['elapsed']]

  test <- farff::readARFF(test_file)
  colnames(test) <- make.names(colnames(test))
  predict_duration <- system.time(prediction())[['elapsed']]

  preds <- preds[c(2:ncol(preds), 1)]
  names(preds)[names(preds) == "response"] <- "predictions"

  if (type == "classification") {
    names(preds) <- sub("^prob.", "", names(preds))
  }
  # FIXME: label encoding for predictions and truth?

  write.csv(preds, file = output_predictions_file, row.names = FALSE)

  meta_results <- data.frame(key=c("training_duration", "predict_duration"),
                             value=c(train_duration, predict_duration))
  write.csv(meta_results, file = meta_results_file, row.names = FALSE)
}

# args = commandArgs(trailingOnly=TRUE)
# run(args[1], args[2], args[3], as.integer(args[4]))

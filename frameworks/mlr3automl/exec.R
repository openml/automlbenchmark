library(mlr3)
library(mlr3tuning)
library(mlr3pipelines)
library(mlr3automl)
library(farff)

run <- function(train_file, test_file, target.index, type, output_predictions_file, cores, time.budget) {
  train <- farff::readARFF(train_file)
  colnames(train) <- make.names(colnames(train))
  target <- colnames(train)[target.index]

  test <- farff::readARFF(test_file)
  colnames(test) <- make.names(colnames(test))

  if (type == "classification") {
    train <- TaskClassif$new("benchmark_train", backend = train, target = target)
    test <- TaskClassif$new("benchmark_test", backend = test, target = target)
  } else if (type == "regression") {
    train <- TaskRegr$new("benchmark_train", backend = train, target = target)
    test <- TaskRegr$new("benchmark_test", backend = test, target = target)
  } else {
    stop("Task type not supported!")
  }

  model <- AutoML(train, terminator = trm("run_time", secs = time.budget - 15))
  model$train()
  preds <- model$predict(test)
  preds <- cbind(preds$data$prob, preds$data$tab)
  names(preds)[names(preds) == "response"] <- "predictions"

  if (type == "classification") {
    names(preds) <- sub("^prob.", "", names(preds))
  }

  preds[, 'row_id'] <- NULL
  write.table(preds, file = output_predictions_file,
    row.names = FALSE, col.names = TRUE,
    sep = ",", quote = FALSE
  )
}

# args = commandArgs(trailingOnly=TRUE)
# run(args[1], args[2], args[3], as.integer(args[4]))

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
    model <- AutoML(train, learner_list = c('classif.xgboost'),
                    terminator = trm('combo', list(trm('run_time', secs = as.integer(time.budget * 0.9)), trm('stagnation', iters = 20))))
  } else if (type == "regression") {
    train <- TaskRegr$new("benchmark_train", backend = train, target = target)
    test <- TaskRegr$new("benchmark_test", backend = test, target = target)
    model <- AutoML(train, learner_list = c('regr.xgboost'),
                    terminator = trm('combo', list(trm('run_time', secs = as.integer(time.budget * 0.9)), trm('stagnation', iters = 20))))
  } else {
    stop("Task type not supported!")
  }

  model$train()
  preds <- model$predict(test)

  if (type == "classification") {
    result = data.frame(preds$data$prob, preds$data$response, preds$data$truth)
    colnames(result) = c(colnames(preds$data$prob), 'predictions', 'truth')
  } else {
    result = data.frame(preds$data$response, preds$data$truth)
    colnames(result) = c('predictions', 'truth')
  }

  write.table(result, file = output_predictions_file,
              row.names = FALSE, col.names = TRUE,
              sep = ",", quote = FALSE
  )
}

# args = commandArgs(trailingOnly=TRUE)
# run(args[1], args[2], args[3], as.integer(args[4]))

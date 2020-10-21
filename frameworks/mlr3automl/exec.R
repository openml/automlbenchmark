library(mlr3)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3tuning)
library(mlr3pipelines)
library(mlr3automl)
library(farff)

run <- function(train_file, test_file, target.index, type, output_predictions_file, cores, time.budget, seed) {
  start_time = Sys.time()
  # AutoML benchmark uses unsigned 32bit integers as seeds, which may be too large for R (32bit signed)
  if (seed > .Machine$integer.max) {
    set.seed(42)
  } else {
    set.seed(seed)
  }
  train <- farff::readARFF(train_file)
  colnames(train) <- make.names(colnames(train))
  target <- colnames(train)[target.index]

  test <- farff::readARFF(test_file)
  colnames(test) <- make.names(colnames(test))

  print(paste("Finished loading data after ", Sys.time() - start_time, " seconds"))
  remaining_budget = as.integer(start_time - Sys.time() + time.budget)
  print(paste("remaining budget: ", remaining_budget, " seconds"))
  if (type == "classification") {
    train <- TaskClassif$new("benchmark_train", backend = train, target = target)
    test <- TaskClassif$new("benchmark_test", backend = test, target = target)
    model <- AutoML(train, learner_timeout = as.integer(remaining_budget * 0.3), resampling = rsmp("holdout"),
                    terminator = trm('combo', list(trm('run_time', secs = as.integer(remaining_budget * 0.9)), trm('stagnation', iters = 20))))
  } else if (type == "regression") {
    train <- TaskRegr$new("benchmark_train", backend = train, target = target)
    test <- TaskRegr$new("benchmark_test", backend = test, target = target)
    model <- AutoML(train, learner_timeout = as.integer(remaining_budget * 0.3), resampling = rsmp("holdout"),
                    terminator = trm('combo', list(trm('run_time', secs = as.integer(remaining_budget * 0.9)), trm('stagnation', iters = 20))))
  } else {
    stop("Task type not supported!")
  }
  print(paste("Finished creating model after ", Sys.time() - start_time, " seconds"))
  model$train()
  print(paste("Finished training model after ", Sys.time() - start_time, " seconds"))
  preds <- model$predict(test)
  print(paste("Finished predictions after ", Sys.time() - start_time, " seconds"))

  if (type == "classification" && !("prob" %in% preds$predict_types)) {
    result = data.frame(preds$data$response, preds$data$truth)
    const_column = rep(0.5, length(preds$response))
    for (level in train$class_names) {
      result = cbind(const_column, result)
    }
    colnames(result) = c(train$class_names, 'predictions', 'truth')
  }
  else if (type == "classification") {
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
  print(paste("Finished writing results after ", Sys.time() - start_time, " seconds"))
}

# args = commandArgs(trailingOnly=TRUE)
# run(args[1], args[2], args[3], as.integer(args[4]))

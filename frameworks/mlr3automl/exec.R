library(mlr3)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3tuning)
library(mlr3pipelines)
library(mlr3automl)
library(mlr3oml)

run <- function(train_file, test_file, target.index, type, output_predictions_file, cores, time.budget, seed, name) {
  start_time = Sys.time()
  # AutoML benchmark uses unsigned 32bit integers as seeds, which may be too large for R (32bit signed)
  if (seed > .Machine$integer.max) {
    set.seed(42)
  } else {
    set.seed(seed)
  }
  train <- mlr3oml::read_arff(train_file)
  colnames(train) <- make.names(colnames(train))
  target <- colnames(train)[target.index]

  test <- mlr3oml::read_arff(test_file)
  colnames(test) <- make.names(colnames(test))
  
  print(paste("Finished loading data after ", Sys.time() - start_time, " seconds"))
  remaining_budget = as.integer(start_time - Sys.time() + time.budget)
  print(paste("remaining budget: ", remaining_budget, " seconds"))
  if (type == "classification") {
    train <- TaskClassif$new("benchmark_train", backend = train, target = target)
    test <- TaskClassif$new("benchmark_test", backend = test, target = target)
    if ("twoclass" %in% train$properties) {
      measure = msr("classif.auc")
    } else {
      measure = msr("classif.logloss")
    }
    model <- AutoML(train, runtime = as.integer(remaining_budget * 0.8), measure = measure)
  } else if (type == "regression") {
    train <- TaskRegr$new("benchmark_train", backend = train, target = target)
    test <- TaskRegr$new("benchmark_test", backend = test, target = target)
    model <- AutoML(train, runtime = as.integer(remaining_budget * 0.8))
  } else {
    stop("Task type not supported!")
  }
  print(paste("Finished creating model after ", difftime(Sys.time(), start_time, units = "secs"), " seconds"))
  model$train()
  print(paste("Finished training model after ", difftime(Sys.time(), start_time, units = "secs"), " seconds"))
  preds <- model$predict(test)
  print(paste("Finished predictions after ", difftime(Sys.time(), start_time, units = "secs"), " seconds"))

  if (type == "classification") {
    sorted_colnames = sort(colnames(preds$data$prob))
    result = data.frame(preds$data$prob[, sorted_colnames], preds$data$response, preds$data$truth)
    colnames(result) = c(sorted_colnames, 'predictions', 'truth')
  } else {
    result = data.frame(preds$data$response, preds$data$truth)
    colnames(result) = c('predictions', 'truth')
  }

  write.table(result, file = output_predictions_file,
              row.names = FALSE, col.names = TRUE,
              sep = ",", quote = FALSE
  )
}

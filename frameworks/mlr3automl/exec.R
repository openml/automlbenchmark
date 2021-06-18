library(mlr3)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3tuning)
library(mlr3pipelines)
library(mlr3automl)
library(mlr3oml)

run <- function(train_file, test_file, target.index, type, output_predictions_file, cores, time.budget, meta_results_file, seed, name) {
  future::plan(future::multicore)
  start_time <- Sys.time()
  set.seed(seed)
  
  train <- mlr3oml::read_arff(train_file)
  colnames(train) <- make.names(colnames(train))
  target <- colnames(train)[target.index]

  test <- mlr3oml::read_arff(test_file)
  colnames(test) <- make.names(colnames(test))
  
  print(paste("Finished loading data after ", Sys.time() - start_time, " seconds"))
  remaining_budget <- as.integer(start_time - Sys.time() + time.budget)
  print(paste("remaining budget: ", remaining_budget, " seconds"))

  if (type == "classification") {
    train <- TaskClassif$new("benchmark_train", backend = train, target = target)
    test <- TaskClassif$new("benchmark_test", backend = test, target = target)
    if ("twoclass" %in% train$properties) {
      measure <- msr("classif.auc")
    } else {
      measure <- msr("classif.logloss")
    }
  } else if (type == "regression") {
    train <- TaskRegr$new("benchmark_train", backend = train, target = target)
    test <- TaskRegr$new("benchmark_test", backend = test, target = target)
    measure <- msr("regr.rmse")
  } else {
    stop("Task type not supported!")
  }

  model <- NULL
  training <- function() {
    model <<- AutoML(train, runtime = as.integer(remaining_budget * 0.8), measure = measure)
    model$train()
  }

  train_duration <- system.time(training())[['elapsed']]
  print(paste("Finished training model after ", difftime(Sys.time(), start_time, units = "secs"), " seconds"))

  preds <- NULL
  prediction <- function() preds <<- model$predict(test)
  predict_duration <- system.time(prediction())[['elapsed']]
  print(paste("Finished predictions after ", difftime(Sys.time(), start_time, units = "secs"), " seconds"))

  if (type == "classification") {
    sorted_colnames <- sort(colnames(preds$data$prob))
    result <- data.frame(preds$data$prob[, sorted_colnames], preds$data$response, preds$data$truth)
    colnames(result) <- c(sorted_colnames, 'predictions', 'truth')
  } else {
    result <- data.frame(preds$data$response, preds$data$truth)
    colnames(result) <- c('presdictions', 'truth')
  }

  write.csv(result, file = output_predictions_file, row.names = FALSE)

  meta_results <- data.frame(key=c("training_duration", "predict_duration"),
                             value=c(train_duration, predict_duration))
  write.csv(meta_results, file = meta_results_file, row.names = FALSE)
}

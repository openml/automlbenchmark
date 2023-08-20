library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(farff)
library(ranger)


run <- function(train_file, test_file, output_predictions_file, cores=1, meta_results_file=NULL, task_type='classification') {
  train <- readARFF(train_file)
  test <- readARFF(test_file)
  colnames(train) <- make.names(colnames(train))
  colnames(test) <- make.names(colnames(test))
  target <- colnames(train)[ncol(train)]
  is_classification <- task_type == "classification"
  task_class <- if (is_classification) TaskClassif else TaskRegr
  train_task <- task_class$new(id="ranger.benchmark.train", backend = train, target = target)
  test_task <- task_class$new(id="ranger.benchmark.test", backend = test, target = target)

  learner <- po("removeconstants") %>>%
         po("imputeoor") %>>%
         po("learner",
            learner = lrn(if (is_classification) "classif.ranger" else "regr.ranger",
                          num.threads = cores,
                          predict_type = if (is_classification) "prob" else "response"))

  mod <- NULL
  preds <- NULL
  training <- function() mod <<- names(learner$train(train_task))
  prediction <- function() preds <<- learner$predict(test_task)[[mod]]
  train_duration <- system.time(training())[['elapsed']]
  predict_duration <- system.time(prediction())[['elapsed']]
  if (is_classification) {
    labels <- colnames(preds$prob)
    as_label <- function(x) labels[[x]]
    predictions <- cbind(preds$prob,
                         predictions=lapply(preds$response, as_label),
                         truth=lapply(preds$truth, as_label)
    )
  } else {
    predictions <- cbind(predictions=preds$response, truth=preds$truth)
  }

  write.csv(predictions, file = output_predictions_file, row.names = FALSE)

  meta_results <- data.frame(key=c("training_duration", "predict_duration"),
                             value=c(train_duration, predict_duration))
  write.csv(meta_results, file = meta_results_file, row.names = FALSE)
}


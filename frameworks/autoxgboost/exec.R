library(mlr)
library(autoxgboost)
library(farff)

run = function(train_file, test_file, output_predictions_file, cores) {
  library(autoxgboost)
  train = farff::readARFF(train_file)
  colnames(train) = make.names(colnames(train))
  target = colnames(train)[ncol(train)]
  train = makeClassifTask(data = train, target = target)

  lrn = makeLearner("classif.autoxgboost", nthread = cores, predict.type = "prob")

  mod = train(lrn, train)

  test = farff::readARFF(test_file)
  colnames(test) = make.names(colnames(test))

  preds = predict(mod, newdata = test)$data
  preds = preds[c(2:ncol(preds), 1)]
  names(preds)[names(preds) == "response"] = "predictions"
  names(preds) = sub("^prob.", "", names(preds))
  # FIXME: label encoding for predictions and truth?

  write.table(preds, file = output_predictions_file,
    row.names = FALSE, col.names = TRUE,
    sep = ",", quote = FALSE
  )
}

# args = commandArgs(trailingOnly=TRUE)
# run(args[1], args[2], args[3], as.integer(args[4]))

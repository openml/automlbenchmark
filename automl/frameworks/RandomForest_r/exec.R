library(mlr)
library(mlrCPO)
library(farff)


run <- function(train_file, test_file, output_predictions_file, cores) {
  train = readARFF(train_file)
  test = readARFF(test_file)
  colnames(train) = make.names(colnames(train))
  colnames(test) = make.names(colnames(test))
  target = colnames(train)[ncol(train)]
  train = makeClassifTask(data = train, target = target)

  lrn = cpoDropConstants() %>>%
        cpoImputeAll(classes = list(numeric = imputeMax(2),
                                    integer = imputeMax(2),
                                    factor = imputeConstant("__MISS__"))
                    ) %>>%
        makeLearner("classif.ranger", num.threads = cores, predict.type = "prob")

  mod = train(lrn, train)
  preds = predict(mod, newdata = test)$data
  preds$truth = NULL

  write.table(preds, file = output_predictions_file,
              row.names = FALSE, col.names = FALSE,
              sep = ",", quote = FALSE)
}

# args = commandArgs(trailingOnly=TRUE)
# run(args[1], args[2], args[3], as.integer(args[4]))

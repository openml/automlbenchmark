library(mlr)
library(mlrCPO)
library(farff)

args = commandArgs(trailingOnly=TRUE)
runtime_seconds = args[1]
number_cores = as.integer(args[2])
performance_metric = args[3]

train = readARFF("/bench/common/train.arff")
test = readARFF("/bench/common/test.arff")
target = colnames(train)[ncol(train)]
train = makeClassifTask(data = train, target = target)

lrn = cpoDropConstants() %>>%
  cpoImputeAll(classes = list(numeric = imputeMax(2),
                              integer = imputeMax(2),
                              factor = imputeConstant("__MISS__"))) %>>%
  makeLearner("classif.ranger", num.threads = number_cores, predict.type = "prob")

mod = train(lrn, train)
preds = predict(mod, newdata = test)$data
preds$truth = NULL

write.table(preds, file = "/bench/common/predictions.csv", row.names = FALSE,
  col.names = FALSE, sep = ",", quote = FALSE)

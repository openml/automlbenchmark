library(readr)
library(BBmisc)

data = read_csv("training.csv", na = c("", "NA", "NULL"))
data$PurchDate = as.numeric(as.POSIXct(as.Date(data$PurchDate, format = "%m/%d/%Y")))
data$RefId = NULL
data = convertDataFrameCols(data, chars.as.factor = TRUE)
data$IsBadBuy = as.factor(data$IsBadBuy)
data$WheelTypeID = as.factor(data$WheelTypeID)
data$IsOnlineSale = as.factor(data$IsOnlineSale)
data$VNZIP1 = as.factor(data$VNZIP1)
data$BYRNO = as.factor(data$BYRNO)
data$WheelTypeID = as.factor(data$WheelTypeID)
summary(data)

library(mlr)
library(mlrCPO)

task = makeClassifTask(id = "kick", data = data, target = "IsBadBuy")

xgb = cpoImpactEncodeClassif() %>>% makeLearner("classif.xgboost", nrounds = 100, predict.type = "prob")


benchmark(list(
  makeLearner("classif.featureless", predict.type = "prob"),
  makeLearner("classif.rpart", predict.type = "prob"),
  xgb),
  task, cv10, logloss)

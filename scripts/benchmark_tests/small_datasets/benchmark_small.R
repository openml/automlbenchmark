library(batchtools)
library(OpenML)
library(mlr)
library(mlrCPO)

LOCAL = TRUE

if (LOCAL) {
  data.ids = c(61)
} else {
  data.ids = c(23381, 6332, 1510, 1515, 470, 40981, 188, 1464, 469, 54, 307, 31, 1494, 1468,
    1068, 1462, 1049, 23, 1050, 1493, 1491, 1492, 40975, 40982, 18, 22, 16, 14, 12, 1067,
    40984, 1487, 40670, 46, 3, 40978, 4134, 40983, 40701, 1489, 1497, 40499, 1475, 4538,
    1053, 4534, 41164, 41146, 41145, 41144, 41143, 41142, 4137, 4136, 1457, 183, 40498,
    41156, 41157, 41158, 41163, 1461, 40900, 1056, 4154
)
}



lrns = list(
  lrn.xgboost = cpoImpactEncodeClassif() %>>%
    makeLearner("classif.xgboost", nrounds = 100, predict.type = "prob"),
  lrn.ranger = cpoImputeAll(classes = list(numeric = imputeHist(), factor = imputeConstant("__MISS__"))) %>>%
    makeLearner("classif.ranger", predict.type = "prob"),
  lrn.rpart = cpoImpactEncodeClassif() %>>%
    makeLearner("classif.rpart", predict.type = "prob"),
  lrn.multinom = cpoImpactEncodeClassif() %>>%
    cpoImputeAll(classes = list(numeric = imputeHist(), factor = imputeConstant("__MISS__"))) %>>%
    makeLearner("classif.multinom", predict.type = "prob"),
  lrn.baseline = makeLearner("classif.featureless", predict.type = "prob")
)

resources = list(
  walltime = 300L,
  memory = 1024 * 2,
  ntasks = 1L,
  ncpus = 1L,
  nodes = 1L,
  clusters = "serial"
  )

reg = makeExperimentRegistry("small_openml", packages = c("OpenML", "mlr", "mlrCPO"))
reg$default.resources = resources


for(data.id in data.ids) {
  d = getOMLDataSet(data.id)
  task = makeClassifTask(id = d$desc$name, data = d$data, target = d$target.features)
  addProblem(name = d$desc$name,
    data = task,
    fun = function(job, data) makeResampleInstance(makeResampleDesc("CV", iters = 5, stratify = TRUE), data))
}

addAlgorithm(name = "mlr",
  fun = function(job, data, instance, lrn) {
    if (length(getTaskClassLevels(data)) == 2)
      measures = list(auc, logloss, acc, timetrain)
    else
      measures = list(multiclass.aunp, logloss, acc, timetrain)
    resample(lrn, data, instance, measures, keep.pred = FALSE)
  }
)


addExperiments(algo.designs = list(mlr = data.table(lrn = lrns)), repls = 1)
summarizeExperiments()

if (LOCAL) {
  submitJobs()
  getStatus()
  x = unwrap(reduceResultsDataTable(fun = function(x) x$aggr))
  res = x[getJobTable()[, algo.pars := sapply(algo.pars, function(x) x$lrn$id)]]
  res
}

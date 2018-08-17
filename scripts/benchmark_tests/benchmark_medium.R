library(batchtools)
library(OpenML)
library(mlr)
library(mlrCPO)

LOCAL = FALSE

if (LOCAL) {
  data.ids = c(61)
  replications = 1L
} else {
  data.ids = c(1112, 1114, 1111, 40996, 40668, 23517, 23512, 4135, 1486, 41027, 1461, 151,
  1590, 41167, 41169, 41168, 41166, 41165, 40685, 41159, 41161, 1216, 23513, 41150,
  41138, 41162)
  replications = 5L
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
  walltime = 3600L,
  memory = 1024L * 62L,
  ntasks = 1L,
  ncpus = 28L,
  nodes = 1L,
  clusters = "mpp2"
  )

reg = makeExperimentRegistry("medium_openml", packages = c("OpenML", "mlr", "mlrCPO"))
reg$default.resources = resources


for(data.id in data.ids) {
  d = getOMLDataSet(data.id)
  task = makeClassifTask(id = d$desc$name, data = d$data, target = d$target.features, fixup.data = "no", check.data = FALSE)
  addProblem(name = d$desc$name,
    data = task,
    fun = function(job, data) makeResampleInstance(makeResampleDesc("Holdout", split = 0.5, stratify = TRUE), data))
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


addExperiments(algo.designs = list(mlr = data.table(lrn = lrns)), repls = replications)
summarizeExperiments()

if (LOCAL) {
  submitJobs()
  getStatus()
  x = unwrap(reduceResultsDataTable(fun = function(x) x$aggr))
  res = x[getJobTable()[, algo.pars := lapply(algo.pars, function(x) x$lrn$id)]]
  res
}

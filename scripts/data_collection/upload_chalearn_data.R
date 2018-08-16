library(rjson)
library(farff)
library(BBmisc)
library(mlr)
library(OpenML)
links = fromJSON(file = "links.json")


for(link in links) {
  print(sprintf("Processing dataset: %s", link$name))
  dir.create(link$name)
  system(sprintf("wget %s -O %s/data.zip", link$link, link$name))
  system(sprintf("cd %s && unzip data.zip", link$name))
  data.train = read.table(sprintf("%s/%s_train.data", link$name, link$name), sep = " ")
  feat.types = read.table(sprintf("%s/%s_feat.type", link$name, link$name))
  categs = which(feat.types[,1] != "Numerical")
  if (length(categs) > 0)
    data.train[, categs] = sapply(data.train[, categs], as.character)
  data.train[, sapply(data.train, is.logical)] = NULL #there is a weird logical param in the last column
  data.train.solution = read.table(sprintf("%s/%s_train.solution", link$name, link$name))
  if (ncol(data.train.solution) > 1)
    data.train.solution = data.frame(x = apply(data.train.solution, 1, function(x) which(as.logical(x)))) - 1
  data = data.frame(class = as.character(data.train.solution[,1]), data.train, stringsAsFactors = FALSE)
  data = convertDataFrameCols(data, chars.as.factor = TRUE)
  data[ is.na(data) ] = NA #convert NaN to NA
  task = makeClassifTask(id = link$name, data = data, target = "class")
  print(task)
  desc = makeOMLDataSetDescription(
    name = link$name,
    description = "The goal of this challenge is to expose the research community to real world datasets of interest to 4Paradigm. All datasets are formatted in a uniform way, though the type of data might differ. The data are provided as preprocessed matrices, so that participants can focus on classification, although participants are welcome to use additional feature extraction procedures (as long as they do not violate any rule of the challenge). All problems are binary classification problems and are assessed with the normalized Area Under the ROC Curve (AUC) metric (i.e. 2*AUC-1).
                   The identity of the datasets and the type of data is concealed, though its structure is revealed. The final score in  phase 2 will be the average of rankings  on all testing datasets, a ranking will be generated from such results, and winners will be determined according to such ranking.
                   The tasks are constrained by a time budget. The Codalab platform provides computational resources shared by all participants. Each code submission will be exceuted in a compute worker with the following characteristics: 2Cores / 8G Memory / 40G SSD with Ubuntu OS. To ensure the fairness of the evaluation, when a code submission is evaluated, its execution time is limited in time.
                   http://automl.chalearn.org/data",
    tag = "automl_chalearn",
    url = link$link,
    creator = "http://automl.chalearn.org",
    default.target.attribute = "class")
  uploadOMLDataSet(task, description = desc, confirm.upload = FALSE)
}


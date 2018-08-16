library(OpenML)
library(tidyverse)
library(moments)

omlcc18 = listOMLTasks(tag = "OpenML-CC18")
oml100 = listOMLTasks(tag = "OpenML100")

omlcc18$omlcc18 = TRUE
oml100$oml100 = TRUE

data = full_join(omlcc18, oml100) %>%
  replace_na(list(omlcc18 = FALSE, oml100 = FALSE))

runs = lapply(data$task.id, function(id) {
  res = list()
  offset = 1
  while (TRUE) {
    res[[length(res) + 1]] = listOMLRunEvaluations(id, offset = offset, limit = 10000, evaluation.measure = "predictive_accuracy")
    offset = offset + 10000
    if (nrow(res[[length(res)]]) == 0) {
      res[[length(res)]] = NULL
      break
    }
  }
  res = do.call(rbind, res)
})

result = do.call(rbind, runs)

result = result %>%
  group_by(task.id, flow.id) %>%
  summarize(mean.perf = mean(predictive.accuracy)) %>%
  ungroup() %>%
  group_by(task.id) %>%
  summarize(n.runs = n(), skewness = skewness(mean.perf)) %>%
  arrange(desc(skewness)) %>%
  full_join(data)

write_csv(result, path = "results/oml_aggregated.csv")

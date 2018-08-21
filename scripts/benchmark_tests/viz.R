library(tidyverse)
library(knitr)
data = readRDS("results.rds")

runtime = data %>%
  mutate(algo.pars = unlist(algo.pars)) %>%
  mutate(algo.pars = fct_relevel(algo.pars, "classif.ranger.impute", "classif.xgboost.impact.encode.classif")) %>%
  select(timetrain.test.mean, algo.pars, problem) %>%
  group_by(algo.pars, problem) %>%
  summarize(mean.runtime = mean(timetrain.test.mean, nr.rm = TRUE)) %>%
  ggplot() +
  geom_col(aes(algo.pars, mean.runtime, fill = algo.pars)) +
  theme_minimal() +
  theme(axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()) +
  facet_wrap(~problem, nrow = 5, scales = "free") +
  theme(legend.position="bottom")

ggsave(runtime, file = "runtime.png", height = 14, width = 14)



data = data %>%
  select(logloss.test.mean, problem, algo.pars) %>%
  mutate(crashed = is.na(logloss.test.mean), algo.pars = unlist(algo.pars)) %>%
  mutate(algo.pars = fct_relevel(algo.pars, "classif.ranger.impute", "classif.xgboost.impact.encode.classif"))


p = data %>%
  ggplot() +
  geom_boxplot(aes(y = logloss.test.mean, x = algo.pars, fill = algo.pars)) +
  theme_minimal() +
  theme(axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()) +
  facet_wrap(~problem, nrow = 5, scales = "free") +
  theme(legend.position="bottom")

ggsave(p, file = "results.png", height = 14, width = 14)



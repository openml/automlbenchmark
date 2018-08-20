library(tidyverse)
data = readRDS("results.rds")

data = data %>%
  select(logloss.test.mean, problem, algo.pars) %>%
  mutate(crashed = is.na(logloss.test.mean), algo.pars = unlist(algo.pars))

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

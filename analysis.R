# generate anytime QD-score plos

library(data.table)
library(ggplot2)
library(pammtools)

# ranger interpretability
x1 = setDT(read.csv("Results/ranger_interpretability_41146.csv"))
x2 = setDT(read.csv("Results/ranger_interpretability_40981.csv"))
x3 = setDT(read.csv("Results/ranger_interpretability_1489.csv"))
x4 = setDT(read.csv("Results/ranger_interpretability_1067.csv"))

x = rbind(x1, x2, x3, x4)

agg = x[, .(mean_qd_score = mean(qd_score), mean_coverage = mean(coverage), mean_obj_max = mean(obj_max), 
            se_qd_score = sd(qd_score) / sqrt(.N), se_coverage = sd(coverage) / sqrt(.N), se_obj_max = sd(obj_max) / sqrt(.N)),
        by = .(iter, method, task)]
agg$method = factor(agg$method, labels = c("MAP-Elites", "CMA-ME", "Gauss.+Imp.", "Random"))
agg$task = factor(agg$task, levels = c("41146", "40981", "1489", "1067"), labels = c("iaml_ranger_41146", "iaml_ranger_40981", "iaml_ranger_1489", "iaml_ranger_1067"))
agg$iter = agg$iter * 100L
final = agg[iter == 100000]

g = ggplot(aes(x = iter, y = mean_qd_score, colour = method, fill = method), data = agg) +
  geom_step() +
  geom_stepribbon(aes(ymin = mean_qd_score - se_qd_score, ymax = mean_qd_score + se_qd_score), colour = NA, alpha = 0.3) + 
  xlab("Evaluations") +
  ylab("QD-Score") +
  labs(colour = "Optimizer", fill = "Optimizer") +
  facet_wrap(~ task, scales = "free") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("Plots/ranger_interpretability.pdf", plot = g, width = 8, height = 6, device = cairo_pdf)

# xgboost interpretability
x1 = setDT(read.csv("Results/xgboost_interpretability_41146.csv"))
x2 = setDT(read.csv("Results/xgboost_interpretability_40981.csv"))
x3 = setDT(read.csv("Results/xgboost_interpretability_1489.csv"))
x4 = setDT(read.csv("Results/xgboost_interpretability_1067.csv"))

x = rbind(x1, x2, x3, x4)

agg = x[, .(mean_qd_score = mean(qd_score), mean_coverage = mean(coverage), mean_obj_max = mean(obj_max), 
            se_qd_score = sd(qd_score) / sqrt(.N), se_coverage = sd(coverage) / sqrt(.N), se_obj_max = sd(obj_max) / sqrt(.N)),
        by = .(iter, method, task)]
agg$method = factor(agg$method, labels = c("MAP-Elites", "CMA-ME", "Gauss.+Imp.", "Random"))
agg$task = factor(agg$task, levels = c("41146", "40981", "1489", "1067"), labels = c("iaml_xgboost_41146", "iaml_xgboost_40981", "iaml_xgboost_1489", "iaml_xgboost_1067"))
agg$iter = agg$iter * 100L
final = agg[iter == 100000]

g = ggplot(aes(x = iter, y = mean_qd_score, colour = method, fill = method), data = agg) +
  geom_step() +
  geom_stepribbon(aes(ymin = mean_qd_score - se_qd_score, ymax = mean_qd_score + se_qd_score), colour = NA, alpha = 0.3) + 
  xlab("Evaluations") +
  ylab("QD-Score") +
  labs(colour = "Optimizer", fill = "Optimizer") +
  facet_wrap(~ task, scales = "free") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("Plots/xgboost_interpretability.pdf", plot = g, width = 8, height = 6, device = cairo_pdf)


# ranger hardware
x1 = setDT(read.csv("Results/ranger_hardware_41146.csv"))
x2 = setDT(read.csv("Results/ranger_hardware_40981.csv"))
x3 = setDT(read.csv("Results/ranger_hardware_1489.csv"))
x4 = setDT(read.csv("Results/ranger_hardware_1067.csv"))

x = rbind(x1, x2, x3, x4)

agg = x[, .(mean_qd_score = mean(qd_score), mean_coverage = mean(coverage), mean_obj_max = mean(obj_max), 
            se_qd_score = sd(qd_score) / sqrt(.N), se_coverage = sd(coverage) / sqrt(.N), se_obj_max = sd(obj_max) / sqrt(.N)),
        by = .(iter, method, task)]
agg$method = factor(agg$method, labels = c("MAP-Elites", "CMA-ME", "Gauss.+Imp.", "Random"))
agg$task = factor(agg$task, levels = c("41146", "40981", "1489", "1067"), labels = c("iaml_ranger_41146", "iaml_ranger_40981", "iaml_ranger_1489", "iaml_ranger_1067"))
agg$iter = agg$iter * 100L
final = agg[iter == 100000]

g = ggplot(aes(x = iter, y = mean_qd_score, colour = method, fill = method), data = agg) +
  geom_step() +
  geom_stepribbon(aes(ymin = mean_qd_score - se_qd_score, ymax = mean_qd_score + se_qd_score), colour = NA, alpha = 0.3) + 
  xlab("Evaluations") +
  ylab("QD-Score") +
  labs(colour = "Optimizer", fill = "Optimizer") +
  facet_wrap(~ task, scales = "free") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("Plots/ranger_hardware.pdf", plot = g, width = 8, height = 6, device = cairo_pdf)


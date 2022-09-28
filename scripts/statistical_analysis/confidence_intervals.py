from statsmodels.stats.proportion import proportion_confint

model_performances = {
  'DNN_DEAP': [71, 100], # [n_of_corrects, n_total]
  'DNN_MAHNOB': [57, 86],
  'CNN_DEAP': [66, 100],
  'CNN_MAHNOB': [56, 86],
}

confidence = 0.95

for model_name, model_performance in model_performances.items():
  lower, upper = proportion_confint(model_performance[0], model_performance[1], 1 - confidence)
  print(f'{model_name}: {(model_performance[0] / model_performance[1]):.3f}, lower={lower:.3f}, upper={upper:.3f}')
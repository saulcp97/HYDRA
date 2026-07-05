#Calibrate optuna experiments
import optuna

def objective(trial):
    #Value to calibrate tetha (float elasticity between 1 and 3)
    frontier = trial.suggest_float('theta', 0., 1.0, log=True)
    #No es elasticidad, es una frontera fija.

    precission = 1
    auc = 1
    return [precission, auc]


study = optuna.create_study(directions=['maximize', "maximize"])
study.optimize(objective, n_trials=5)
print("Best Hyperparameters:", study.best_params)
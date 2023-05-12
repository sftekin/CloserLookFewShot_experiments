import optuna


# define objective function to be optimized.
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


def run():
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print(study.best_params)


if __name__ == '__main__':
    run()


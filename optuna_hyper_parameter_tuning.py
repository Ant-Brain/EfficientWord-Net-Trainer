from simple_model import CLASSES, AttentiveMobileWordClassifierPL, AudioClassifierVectorDatasetPL
import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything

MAX_EPOCHS=1

def objective(trial:optuna.trial.Trial):

    s = trial.suggest_float("scale", 2.0 , 10)
    m = trial.suggest_float("margin", -0.2, 0.2)
    lr = trial.suggest_categorical("lr", [1e-2,1e-3])

    print(f"Scale :{s}")
    print(f"Margin:{m}")
    print(f"LR    :{lr}")

    seed_everything(69)

    pl_model = AttentiveMobileWordClassifierPL(class_count = len(CLASSES), lr= lr)
    pl_data_module = AudioClassifierVectorDatasetPL()

    pl_model.pytorch_model.update_margin(m , s)
    trainer = pl.Trainer(
        precision=16,
        max_epochs = MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(pl_model, pl_data_module)
    print("bruhhhh")
    print(trainer.callback_metrics)
    return trainer.callback_metrics["test_top10"].item()

pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(direction="maximize", pruner = pruner)
study.optimize(objective, n_trials=100,)

print("Best Trial")
trial = study.best_trial
print("Params: ")
vals = dict()
for key, value in trial.params.items() :
    print(f"    {key}: {value}")
    vals[key]=value

import json
open("best_vals.json", 'w').write(json.dumps(vals, indent=4))
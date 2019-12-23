import wandb

from wandb.sweeps.config import tune
from wandb.sweeps.config.hyperopt import hp
from wandb.sweeps.config.tune.suggest.hyperopt import HyperOptSearch


tune_config = tune.run(
    'run_experiment.py',
    search_alg=HyperOptSearch({
            'safety_gamma': hp.uniform('safety_gamma', 0.05, 0.1),
            'target_entropy': hp.choice('target_entropy', [-16, -8, -4, -2]),
            'batch_size': hp.choice('batch_size', [32, 256]),
            'reward_scale_factor': hp.uniform('reward_scale_factor', 0, 2)},
        metric='Metrics/AverageReturn',
        mode='max'),
    num_samples=10,
)

tune_config.save('sweep-tune-hyperopt.yaml')

sweep_id = wandb.sweep(tune_config, entity='krshna', project='safemrl')
# wandb.agent(sweep_id, run_experiment.train)
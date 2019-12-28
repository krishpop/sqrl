import numpy as np
import wandb

from wandb.sweeps.config import tune
from wandb.sweeps.config.hyperopt import hp
from wandb.sweeps.config.tune.suggest.hyperopt import HyperOptSearch


tune_config = tune.run(
    'run_experiment.py',
    search_alg=HyperOptSearch({
            'target_entropy': hp.choice('target_entropy', [-16, -8, -4, -2]),
            'actor_lr': hp.uniform('actor_lr', -5, -3),
            'critic_lr': hp.uniform('critic_lr', -5, -3),
            'entropy_lr': hp.uniform('entropy_lr', -5, -3),
            'reward_scale_factor': hp.uniform('reward_scale_factor', 0, 2),
            'initial_log_alpha': hp.uniform('initial_log_alpha', -2, 2)},
        metric='AverageReturn',
        mode='max',
        max_concurrent=4,
        points_to_evaluate=[{
          'target_entropy': -8,
          'actor_lr': -3.5,
          'critic_lr': -3.5,
          'entropy_lr': -3.5,
          'reward_scale_factor': 1.,
          'initial_log_alpha': 0
        }]),
    num_samples=20)

tune_config.save('sac-sweep-tune-hyperopt.yaml')

sweep_id = wandb.sweep(tune_config, entity='krshna', project='safemrl')
# wandb.agent(sweep_id, run_experiment.train)

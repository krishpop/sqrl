import numpy as np
import wandb

from wandb.sweeps.config import tune
from wandb.sweeps.config.hyperopt import hp
from wandb.sweeps.config.tune.suggest.hyperopt import HyperOptSearch


tune_config = tune.run(
    'run_experiment.py',
    search_alg=HyperOptSearch({
          'actor_lr': hp.uniform('actor_lr', -5, -4),
          'critic_lr': hp.uniform('critic_lr', -4, -3),
          'reward_scale_factor': hp.uniform('reward_scale_factor', 0.1, 2),
          'layer_size': hp.choice('layer_size', [32, 64, 128, 256]),
          'gradient_clipping': hp.uniform('gradient_clipping', 0., 4.)
        },
        metric='AverageReturn',
        mode='max',
        max_concurrent=4,
        points_to_evaluate=[{
          'target_entropy': -24,
          'actor_lr': 1e-4,
          'critic_lr': 1e-3,
          'reward_scale_factor': 0.3
        }]),
    num_samples=20)

tune_config.save('wcpg-ls-gc-sweep-tune-hyperopt.yaml')

sweep_id = wandb.sweep(tune_config, entity='krshna', project='safemrl-2')
# wandb.agent(sweep_id, run_experiment.train)

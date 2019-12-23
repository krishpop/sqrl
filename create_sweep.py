import wandb

from wandb.sweeps.config import tune
from wandb.sweeps.config.hyperopt import hp
from wandb.sweeps.config.tune.suggest.hyperopt import HyperOptSearch


tune_config = tune.run(
    'run_experiment.py',
    search_alg=HyperOptSearch({
            'target_entropy': hp.choice('target_entropy', [-16, -8, -4, -2]),
            'actor_lr': hp.loguniform(1e-5, 1e-3, base=10),
            'critic_lr': hp.loguniform(1e-5, 1e-3, base=10),
            'entropy_lr': hp.loguniform(1e-5, 1e-3, base=10),
            'reward_scale_factor': hp.uniform('reward_scale_factor', 0, 2),
            'initial_log_alpha': hp.uniform('initial_log_alpha', -2, 2)},
        metric='Metrics/AverageReturn',
        mode='max',
        gin_files=['minitaur_default.gin', 'sac.gin', 'networks.gin'],
        env_str='MinitaurRandFrictionGoalVelocityEnv-v0',
        lr=None),
    num_samples=10,
)

tune_config.save('sac-sweep-tune-hyperopt.yaml')

sweep_id = wandb.sweep(tune_config, entity='krshna', project='safemrl')
# wandb.agent(sweep_id, run_experiment.train)
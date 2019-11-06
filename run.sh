for AC_LR in 1e-5 1e-4 1e-4
do
    python safemrl/algos/train.py --root_dir=../tfagents/baselines/ensemble-1 --gin_file=safemrl/configs/sac_ensemble.gin \
    --gin_file=safemrl/configs/minitaur_default.gin --gin_param="train_eval_ensemble.train_eval.actor_learning_rate = ${AC_LR}"
done
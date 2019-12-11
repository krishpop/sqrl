
for SG in 0.5 0.6 0.7
do
    python safemrl/algos/train.py --root_dir=$EXP_DIR/baselines/sg-$SG \
        --gin_file=point_mass_default.gin --gin_file=sac_safe_online.gin \
        --gin_file=networks.gin --gin_param="safe_sac_agent.SafeSacAgentOnline.safety_gamma = ${SG}"
done
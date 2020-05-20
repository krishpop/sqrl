for SG in 0.6 0.7 0.8
do
	for TS in 0.1 0.05 0.125
	do
	    python train.py --root_dir=$EXP_DIR/baselines/collect-sg-${SG}-ts-${TS} \
		--gin_file=point_mass_default.gin --gin_file=sac_safe_online.gin \
		--gin_file=networks_tiny.gin --gin_param="safe_sac_agent.SqrlAgent.safety_gamma = ${SG}" \
		--gin_param="safe_sac_agent.SqrlAgent.target_safety = ${TS}" \
		--gin_param="ENV_STR = 'DrunkSpiderShort-acnoise'" &> ~/outs/collect-sg-${SG}-ts-${TS}.out&
	done
done

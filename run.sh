
python train.py --root_dir=$EXP_DIR/$1 --gin_file=$CONFIG_DIR/sac_safe_online.gin \
    --gin_file=$CONFIG_DIR/point_mass_default.gin --gin_file=$CONFIG_DIR/networks_tiny.gin

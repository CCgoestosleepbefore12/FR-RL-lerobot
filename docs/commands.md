
pickup task

--without backup

1.
python scripts/real/deploy_bc_inference.py --ckpt checkpoints/no_bias/pickup/pickup_bc_iter2_20260430_155321/checkpoints/020000/pretrained_model --task pickup --lift-threshold 0.04

2.
python scripts/real/deploy_bc_inference.py --ckpt checkpoints/no_bias/pickup/pickup_bc_iter2_20260430_155321/checkpoints/020000/pretrained_model --task pickup --lift-threshold 0.04 --bias --bias-range -0.2 0.2 --bias-monitor


--with backup

1.--no bias
python scripts/real/deploy_pickup_with_backup.py --bc-ckpt checkpoints/no_bias/pickup/pickup_bc_iter2_20260430_155321/checkpoints/020000/pretrained_model --ckpt-version v3 --lift-threshold 0.04 

2.--with bias
python scripts/real/deploy_pickup_with_backup.py --bc-ckpt checkpoints/no_bias/pickup/pickup_bc_iter2_20260430_155321/checkpoints/020000/pretrained_model --ckpt-version v3 --lift-threshold 0.04 --bias --bias-range -0.1 0.1 --bias-monitor

python scripts/real/deploy_pickup_with_backup.py --bc-ckpt checkpoints/no_bias/pickup/pickup_bc_iter2_20260430_155321/checkpoints/020000/pretrained_model --ckpt-version v3 --lift-threshold 0.04 --bias --bias-range -0.2 0.2 --bias-monitor


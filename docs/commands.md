上手清单（pickup 真机从零到 iter2）                                         
                                                            
  # 1. 收 50 条 cold-start demo                                               
  python scripts/real/collect_demo_task_policy.py --task pickup -n 50         
                                                                              
  # 2. cold-start BC                                                          
   python scripts/tools/bc_pretrain_task_policy.py \
      --config scripts/configs/train_hil_sac_pickup_real.json \
      --demo-paths "data/with_bias/pickup/pickup_demos_*.pkl" \               
      --steps 20000 \
      --output-dir checkpoints/with_bias/pickup/pickup_bc_$(date+%Y%m%d_%H%M%S)    
                                                                              
  # 3. 部署测一下，记新 ckpt 路径                                             
  python scripts/real/deploy_bc_inference.py \                                
      --ckpt <step3 出的路径>/checkpoints/020000/pretrained_model \           
      --task pickup --lift-threshold 0.04                                     
                         /home/lab1/FR-RL-lerobot/checkpoints/with_bias/pickup/pickup_bc_20260507_163556/checkpoints/020000/pretrained_model                                                     
  # 4. 收 30 条 dagger iter1（基于 cold-start ckpt）                          
  python scripts/real/deploy_bc_with_dagger.py \                              
      --ckpt <step3 出的路径>/checkpoints/020000/pretrained_model \           
      --task pickup --iter 1 -n 30                                            
   
  # 5. 训 iter1（demos + dagger 介入帧合并）                                   
   python scripts/tools/bc_pretrain_task_policy.py \
      --config scripts/configs/train_hil_sac_pickup_real.json \
      --demo-paths "data/with_bias/pickup/*.pkl" \
      --steps 20000 --intervention-only \
      --output-dir checkpoints/with_bias/pickup/pickup_bc_iter1_$(date
  +%Y%m%d_%H%M%S)
                                                         
                                                            
  # 6. 重复 4-6 拿 iter2、iter3... 



上手清单 — With bias 版本                                                   
                                         
  # 1. 收 50 条 cold-start demo（带 bias）                                    
  python scripts/real/collect_demo_task_policy.py \                           
      --task pickup -n 50 \                                 
      --bias --bias-range -0.2 0.2                  
   
  # 2. cold-start BC                                                          
  python scripts/tools/bc_pretrain_task_policy.py \         
      --config scripts/configs/train_hil_sac_pickup_real.json \                
      --demo-paths "data/with_bias/pickup/pickup_demos_*.pkl" \
      --steps 20000 \                                                         
      --output-dir checkpoints/with_bias/pickup/pickup_bc_$(date
  +%Y%m%d_%H%M%S)                                                             
                                                            
  # 3. 部署测一下（带 bias，跟训练分布一致）                                  
  python scripts/real/deploy_bc_inference.py \
      --ckpt <step3 出的路径>/checkpoints/020000/pretrained_model \           
      --task pickup --lift-threshold 0.04 \                 
      --bias --bias-range -0.2 0.2
      --bias-monitor                                                     
                                                            
  # 4. 收 30 条 dagger iter1（带 bias）                                       
  python scripts/real/deploy_bc_with_dagger.py \            
      --ckpt <step3 出的路径>/checkpoints/020000/pretrained_model \           
      --task pickup --iter 1 -n 30 --bias --bias-range -0.2 0.2
                                                                              
  # 5. 训 iter1（demos + dagger 介入帧合并）                              
    python scripts/tools/bc_pretrain_task_policy.py \
      --config scripts/configs/train_hil_sac_pickup_real.json \
      --demo-paths "data/with_bias/pickup/*.pkl" \
      --steps 20000 --intervention-only \
      --output-dir checkpoints/with_bias/pickup/pickup_bc_iter1_$(date +%Y%m%d_%H%M%S)
                                 
                                                            
  # 7. 重复 4-6 拿 iter2、iter3...     



union deploy:

 python scripts/real/deploy_pickup_with_backup.py 
      --bc-ckpt checkpoints/with_bias/pickup/pickup_bc_iter1_20260508_133945/checkpoints/020000/pretrained_model 
      --ckpt-version v3 --lift-threshold 0.04 
      --bias --bias-range -0.2 0.2 --bias-monitor

python scripts/real/deploy_pickup_with_backup.py 
      --bc-ckpt checkpoints/with_bias/pickup/pickup_bc_iter1_20260508_133945/checkpoints/020000/pretrained_model 
      --ckpt-version v3 --lift-threshold 0.04




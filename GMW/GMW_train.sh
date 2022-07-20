# CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch --master_port 33521 --nproc_per_node=4 \
main.py --log-dir ./logs/big_demo \
-b 8 --lr 1e-4 --epoch 100 --val_freq 5 \
--train_data_path /mnt/truenas/scratch/yingyan.li/Pairs_Depth/output/kitti/pairs/sync_with_corner_loss_eval_1500_0/gen_data_train.json \
--val_data_path /mnt/truenas/scratch/yingyan.li/Pairs_Depth/output/kitti/pairs/sync_with_corner_loss_eval_1500_0/gen_data_infer.json

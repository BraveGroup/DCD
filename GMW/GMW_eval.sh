python -m torch.distributed.launch --master_port 27561 --nproc_per_node=4 \
main.py --log-dir ./logs/debug \
-b 8 --lr 1e-4 --epoch 100 --val_freq 5 \
-e \
--resume logs/GMW/checkpoint_epoch_100.pth.tar
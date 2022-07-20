CUDA_VISIBLE_DEVICES=4,5 \
python tools/plain_train_net.py --batch_size 8 --config runs/DGDE.yaml \
--output output/DGDE --num_gpus 2 \
# --generate_for_GMW \
# --ckpt output/DGDE/model_final.pth

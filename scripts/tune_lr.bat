# 学习率粗筛示例（在 C:\Users\cc\Desktop\1 下执行）
# 约 40 epoch × 3 个 lr，总时间大约数小时

python 1/tune_lr.py ^
  --lrs 0.0005,0.001,0.002 ^
  --nb_epochs 40 ^
  --val_iter 5 ^
  --weight 0.5 ^
  --fold_id 0 ^
  --preprocessed 1

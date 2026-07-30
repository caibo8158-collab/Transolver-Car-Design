export CUDA_VISIBLE_DEVICES=0

python main.py \
--cfd_model=Transolver \
--data_dir "C:/Users/cc/Downloads/mlcfd_data/mlcfd_data/training_data" \
--save_dir "C:/Users/cc/Downloads/mlcfd_data/preprocessed_data" \
--preprocessed 1

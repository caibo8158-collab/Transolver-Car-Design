export CUDA_VISIBLE_DEVICES=0

python main_evaluation.py \
--cfd_model=Transolver \
--nb_epochs=200 \
--weight=0.5 \
--data_dir "C:/Users/cc/Downloads/mlcfd_data/mlcfd_data/training_data" \
--save_dir "C:/Users/cc/Downloads/mlcfd_data/preprocessed_data" \

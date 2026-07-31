import train
import os
import torch
import argparse

from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset
from models.Transolver import Model

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser()
# 原始 CFD 数据目录（含 param0~param8）
parser.add_argument('--data_dir', default=r'C:\Users\cc\Downloads\mlcfd_data\mlcfd_data\training_data')
# 预处理结果保存目录（x.npy / y.npy 等）
parser.add_argument('--save_dir', default=r'C:\Users\cc\Downloads\mlcfd_data\preprocessed_data')
# 交叉验证折编号：该折作验证集，其余作训练集
parser.add_argument('--fold_id', default=0, type=int)
# 使用的 GPU 编号
parser.add_argument('--gpu', default=0, type=int)
# 每隔多少个 epoch 做一次验证
parser.add_argument('--val_iter', default=10, type=int)
parser.add_argument('--cfd_config_dir', default='cfd/cfd_params.yaml')
# 模型名称，如 Transolver
parser.add_argument('--cfd_model', default='Transolver')
# 是否使用 CFD 网格构图（不加该 flag 则为 False）
parser.add_argument('--cfd_mesh', action='store_true')
# 构图时的邻域半径
parser.add_argument('--r', default=0.2, type=float)
# 压力损失权重：总损失 = 速度损失 + weight * 压力损失
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--nb_epochs', default=200, type=int)
# 1：假定 save_dir 已有预处理 npy；0：从 VTK 现场预处理并保存（首次运行请用 0）
parser.add_argument('--preprocessed', default=1, type=int)
# global: 全体点统一 mean/std；separate: 表面/体积点分开归一化
parser.add_argument('--norm_mode', default='global', choices=['global', 'separate'])
args = parser.parse_args()
print(args)

# 传给 train.main 的训练超参数
hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}

# ---------- 设备 ----------
n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

# ---------- 数据 ----------
# 按 fold 划分训练/验证并加载；返回:
#   train_data: 训练集样本列表（每个为 PyG Data，含 x/y/pos/surf/edge_index）
#   val_data:   验证集样本列表
#   coef_norm:  归一化系数 (mean_in, std_in, mean_out, std_out)
train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=args.preprocessed)
# 封装为图数据集（点特征、边等）
train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

# ---------- 模型 ----------
if args.cfd_model == 'Transolver':
    # space_dim=7：坐标+SDF+法向等输入；out_dim=4：3 维速度 + 1 维压力
    # slice_num：Physics-Attention 的切片数
    model = Model(n_hidden=256, n_layers=8, space_dim=7,
                  fun_dim=0,
                  n_head=8,
                  mlp_ratio=2, out_dim=4,
                  slice_num=32,
                  unified_pos=0).to(device)
else:
    raise ValueError(f'未知模型: {args.cfd_model!r}，目前仅支持 Transolver')

# 日志与权重保存路径
path = f'metrics/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}_{args.norm_mode}'
if not os.path.exists(path):
    os.makedirs(path)

# ---------- 开始训练 ----------
# train.main 输入:
#   device:     训练设备（cuda:i 或 cpu）
#   train_ds:   训练集 GraphDataset，get → (cfd_data, shape)
#   val_ds:     验证集 GraphDataset
#   model:      已构建的 Transolver（7 维点特征 → 4 维速度+压力）
#   hparams:    {'lr', 'batch_size', 'nb_epochs'}
#   path:       日志与权重保存目录
#   val_iter:   每隔多少 epoch 验证一次
#   reg:        压力损失权重（即 args.weight）；总损失 = 速度损失 + reg * 压力损失
#   coef_norm:  归一化系数，供日志记录 / 后续反归一化
# 返回: 训练完成后的模型
model = train.main(device, train_ds, val_ds, model, hparams, path, val_iter=args.val_iter, reg=args.weight,
                   coef_norm=coef_norm)

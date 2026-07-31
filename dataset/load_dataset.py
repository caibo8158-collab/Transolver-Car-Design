import os
from dataset.dataset import get_datalist


def get_samples(root):
    """
    扫描 data_dir 下 param0~param8，收集每个折里的样本相对路径。

    返回:
        samples: list[list[str]]，长度 9；
                 samples[i] 为第 i 折的样本列表，元素形如 'param0/车型ID'
    """
    folds = [f'param{i}' for i in range(9)]
    samples = []
    for fold in folds:
        fold_samples = []
        files = os.listdir(os.path.join(root, fold))
        for file in files:
            path = os.path.join(root, os.path.join(fold, file))
            # 只收子目录（真正的车样本），跳过 Cd.npy、param.py 等文件
            if os.path.isdir(path):
                fold_samples.append(os.path.join(fold, file))
        samples.append(fold_samples)
    return samples  # 100 + 99 + 97 + 100 + 100 + 96 + 100 + 98 + 99 = 889 samples


def load_train_val_fold(args, preprocessed):
    """
    按 fold_id 做交叉验证划分，并加载训练/验证数据。

    - fold_id 对应折 → 验证集
    - 其余 8 折 → 训练集
    - 训练集: norm=True，会计算归一化系数 coef_norm
    - 验证集: 使用同一套 coef_norm，保证标准化一致
    - preprocessed: True 读 npy；False 从 VTK 预处理并缓存到 save_dir

    供 main.py 训练使用。
    返回: train_dataset, val_dataset, coef_norm
    """
    samples = get_samples(args.data_dir)
    trainlst = []
    for i in range(len(samples)):
        if i == args.fold_id:
            continue
        trainlst += samples[i]
    vallst = samples[args.fold_id] if 0 <= args.fold_id < len(samples) else None

    norm_mode = getattr(args, 'norm_mode', 'global')
    if preprocessed:
        print("use preprocessed data")
    print("loading data, norm_mode=", norm_mode)
    # 训练集：同时返回 dataset 与归一化系数
    train_dataset, coef_norm = get_datalist(
        args.data_dir, trainlst, norm=True, savedir=args.save_dir,
        preprocessed=preprocessed, norm_mode=norm_mode)
    # 验证集：传入 coef_norm，不再重新统计均值方差
    val_dataset = get_datalist(
        args.data_dir, vallst, coef_norm=coef_norm, savedir=args.save_dir,
        preprocessed=preprocessed, norm_mode=norm_mode)
    print("load data finish")
    return train_dataset, val_dataset, coef_norm


def load_train_val_fold_file(args, preprocessed):
    """
    与 load_train_val_fold 基本相同，额外返回 vallst（验证样本路径列表）。

    供 main_evaluation.py 评估使用：需要样本路径去算阻力系数等。
    返回: train_dataset, val_dataset, coef_norm, vallst
    """
    samples = get_samples(args.data_dir)
    trainlst = []
    for i in range(len(samples)):
        if i == args.fold_id:
            continue
        trainlst += samples[i]
    vallst = samples[args.fold_id] if 0 <= args.fold_id < len(samples) else None

    norm_mode = getattr(args, 'norm_mode', 'global')
    if preprocessed:
        print("use preprocessed data")
    print("loading data, norm_mode=", norm_mode)
    train_dataset, coef_norm = get_datalist(
        args.data_dir, trainlst, norm=True, savedir=args.save_dir,
        preprocessed=preprocessed, norm_mode=norm_mode)
    val_dataset = get_datalist(
        args.data_dir, vallst, coef_norm=coef_norm, savedir=args.save_dir,
        preprocessed=preprocessed, norm_mode=norm_mode)
    print("load data finish")
    return train_dataset, val_dataset, coef_norm, vallst

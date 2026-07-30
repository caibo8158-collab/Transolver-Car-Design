"""
学习率超参数搜索（粗筛）：
- 数据只加载一次
- 每个学习率重新初始化模型，短训若干 epoch
- 用验证集最优 / 最终 val_loss 选出最佳 lr
- 结果保存到 metrics/lr_tune/

示例（在 Desktop\\1 下运行，以便 metrics 路径一致）：
  python 1/tune_lr.py --lrs 0.0005,0.001,0.002 --nb_epochs 40 --val_iter 5
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

import train
from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset
from models.Transolver import Model


def parse_lrs(text):
    vals = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError('至少提供一个学习率，例如 --lrs 0.0005,0.001,0.002')
    return vals


def lr_tag(lr):
    """把 0.001 转成文件名友好的 0p001。"""
    return f'{lr:.6f}'.rstrip('0').rstrip('.').replace('.', 'p')


def build_model(device):
    return Model(
        n_hidden=256,
        n_layers=8,
        space_dim=7,
        fun_dim=0,
        n_head=8,
        mlp_ratio=2,
        out_dim=4,
        slice_num=32,
        unified_pos=0,
    ).to(device)


def summarize_history(history, lr, time_elapsed=None):
    val_losses = history.get('val_loss', [])
    val_epochs = history.get('val_epoch', [])
    train_losses = history.get('train_loss', [])
    if not val_losses:
        best_val, best_epoch, final_val = None, None, None
    else:
        idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
        best_val = val_losses[idx]
        best_epoch = val_epochs[idx]
        final_val = val_losses[-1]
    return {
        'lr': lr,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': final_val,
        'best_val_loss': best_val,
        'best_val_epoch': best_epoch,
        'time_elapsed': time_elapsed,
    }


def plot_lr_comparison(all_histories, save_path):
    plt.figure(figsize=(9, 5))
    for lr, hist in all_histories:
        if hist.get('val_epoch'):
            plt.plot(
                hist['val_epoch'],
                hist['val_loss'],
                marker='o',
                markersize=3,
                linewidth=1.5,
                label=f'lr={lr:g}',
            )
    plt.xlabel('Epoch')
    plt.ylabel('val_loss')
    plt.title('Learning Rate Sweep: Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Comparison curve saved to: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Learning rate sweep for Transolver')
    parser.add_argument('--data_dir', default=r'C:\Users\cc\Downloads\mlcfd_data\mlcfd_data\training_data')
    parser.add_argument('--save_dir', default=r'C:\Users\cc\Downloads\mlcfd_data\preprocessed_data')
    parser.add_argument('--fold_id', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--val_iter', default=5, type=int, help='短训建议 5，多采几个 val 点')
    parser.add_argument('--cfd_model', default='Transolver')
    parser.add_argument('--cfd_mesh', action='store_true')
    parser.add_argument('--r', default=0.2, type=float)
    parser.add_argument('--weight', default=0.5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--nb_epochs', default=40, type=int, help='粗筛轮数，不必 200')
    parser.add_argument('--preprocessed', default=1, type=int)
    parser.add_argument(
        '--lrs',
        default='0.0005,0.001,0.002',
        help='逗号分隔的学习率列表，例如 0.0005,0.001,0.002',
    )
    parser.add_argument(
        '--out_root',
        default='metrics/lr_tune',
        help='调参结果根目录（相对当前工作目录）',
    )
    args = parser.parse_args()
    lrs = parse_lrs(args.lrs)
    print('LR candidates:', lrs)
    print(args)

    n_gpu = torch.cuda.device_count()
    use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    print('device:', device)

    # 数据只准备一次，后面每个 lr 复用
    train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=args.preprocessed)
    train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
    val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

    sweep_dir = os.path.join(
        args.out_root,
        args.cfd_model,
        str(args.fold_id),
        f'ep{args.nb_epochs}_w{args.weight}',
    )
    os.makedirs(sweep_dir, exist_ok=True)

    results = []
    all_histories = []

    for lr in lrs:
        tag = lr_tag(lr)
        run_dir = os.path.join(sweep_dir, f'lr{tag}')
        os.makedirs(run_dir, exist_ok=True)
        print('\n' + '=' * 60)
        print(f'Training with lr={lr:g}  ->  {run_dir}')
        print('=' * 60)

        model = build_model(device)
        hparams = {'lr': lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}
        train.main(
            device,
            train_ds,
            val_ds,
            model,
            hparams,
            run_dir,
            val_iter=args.val_iter,
            reg=args.weight,
            coef_norm=coef_norm,
        )

        hist_path = os.path.join(run_dir, f'loss_history_{args.nb_epochs}.json')
        with open(hist_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        log_path = os.path.join(run_dir, f'log_{args.nb_epochs}.json')
        time_elapsed = None
        if os.path.exists(log_path):
            # log 可能因多次 append 不是单一 JSON，尽量读最后一段；失败则忽略耗时
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    raw = f.read().strip()
                # 取最后一个完整对象
                last = raw.rfind('{')
                if last >= 0:
                    time_elapsed = json.loads(raw[last:]).get('time_elapsed')
            except Exception:
                time_elapsed = None

        summary = summarize_history(history, lr, time_elapsed=time_elapsed)
        results.append(summary)
        all_histories.append((lr, history))
        print('Summary:', summary)

        # 释放显存，准备下一组 lr
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 按 best_val_loss 排序（越小越好）
    ranked = sorted(
        [r for r in results if r['best_val_loss'] is not None],
        key=lambda x: x['best_val_loss'],
    )
    best = ranked[0] if ranked else None

    report = {
        'settings': {
            'lrs': lrs,
            'nb_epochs': args.nb_epochs,
            'val_iter': args.val_iter,
            'weight': args.weight,
            'fold_id': args.fold_id,
            'batch_size': args.batch_size,
            'r': args.r,
        },
        'results': results,
        'ranking_by_best_val_loss': ranked,
        'recommended_lr': best['lr'] if best else None,
    }
    report_path = os.path.join(sweep_dir, 'lr_sweep_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport saved to: {report_path}')

    plot_path = os.path.join(sweep_dir, 'lr_sweep_val_curves.png')
    plot_lr_comparison(all_histories, plot_path)

    print('\n===== LR Sweep Ranking (best_val_loss ascending) =====')
    for i, r in enumerate(ranked, 1):
        print(
            f"{i}. lr={r['lr']:g}  best_val={r['best_val_loss']:.6f} "
            f"(epoch {r['best_val_epoch']})  final_val={r['final_val_loss']:.6f}"
        )
    if best:
        print(f'\nRecommended learning rate: {best["lr"]:g}')
        print('Next step: 用该 lr 做更长训练，例如')
        print(
            f'  python 1/main.py --lr {best["lr"]} --nb_epochs 160 --weight {args.weight} --preprocessed 1'
        )


if __name__ == '__main__':
    main()

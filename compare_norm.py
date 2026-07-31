"""
对比两种归一化：
  - global:   全体点统一 mean/std（原版）
  - separate: 表面点 / 体积点分开统计并归一化

各训 nb_epochs（默认 40），以物理量纲相对 L2（反归一化后）公平对比。

示例:
  python compare_norm.py --nb_epochs 40 --val_iter 5
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

import train
from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset, coef_norm_to_jsonable
from models.Transolver import Model


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


def plot_comparison(results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for r in results:
        hist = r['history']
        if hist.get('val_epoch'):
            ax.plot(hist['val_epoch'], hist['val_l2_press'], marker='o', markersize=3,
                    linewidth=1.5, label=r['norm_mode'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('val relative L2 (pressure)')
    ax.set_title('Pressure L2 (denormalized)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for r in results:
        hist = r['history']
        if hist.get('val_epoch'):
            ax.plot(hist['val_epoch'], hist['val_l2_velo'], marker='o', markersize=3,
                    linewidth=1.5, label=r['norm_mode'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('val relative L2 (velocity)')
    ax.set_title('Velocity L2 (denormalized)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle('Normalization Comparison')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Comparison curve saved to: {save_path}')


def summarize(norm_mode, history):
    def last(key):
        vals = history.get(key) or []
        return vals[-1] if vals else None

    def best(key):
        vals = history.get(key) or []
        epochs = history.get('val_epoch') or []
        if not vals:
            return None, None
        i = min(range(len(vals)), key=lambda k: vals[k])
        return vals[i], epochs[i]

    best_p, ep_p = best('val_l2_press')
    best_v, ep_v = best('val_l2_velo')
    return {
        'norm_mode': norm_mode,
        'final_train_loss': last('train_loss'),
        'final_val_loss': last('val_loss'),
        'final_val_l2_press': last('val_l2_press'),
        'final_val_l2_velo': last('val_l2_velo'),
        'best_val_l2_press': best_p,
        'best_val_l2_press_epoch': ep_p,
        'best_val_l2_velo': best_v,
        'best_val_l2_velo_epoch': ep_v,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare global vs separate normalization')
    parser.add_argument('--data_dir', default=r'C:\Users\cc\Downloads\mlcfd_data\mlcfd_data\training_data')
    parser.add_argument('--save_dir', default=r'C:\Users\cc\Downloads\mlcfd_data\preprocessed_data')
    parser.add_argument('--fold_id', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--val_iter', default=5, type=int)
    parser.add_argument('--cfd_model', default='Transolver')
    parser.add_argument('--cfd_mesh', action='store_true')
    parser.add_argument('--r', default=0.2, type=float)
    parser.add_argument('--weight', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--nb_epochs', default=40, type=int)
    parser.add_argument('--preprocessed', default=1, type=int)
    parser.add_argument(
        '--modes',
        default='global,separate',
        help='逗号分隔，默认 global,separate',
    )
    parser.add_argument('--out_root', default='metrics/norm_compare')
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    print('Modes:', modes)
    print(args)

    n_gpu = torch.cuda.device_count()
    use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    print('device:', device)

    sweep_dir = os.path.join(
        args.out_root,
        args.cfd_model,
        str(args.fold_id),
        f'ep{args.nb_epochs}_w{args.weight}_lr{args.lr}',
    )
    os.makedirs(sweep_dir, exist_ok=True)

    results = []
    summaries = []

    for mode in modes:
        print('\n' + '=' * 60)
        print(f'Running norm_mode={mode}')
        print('=' * 60)
        args.norm_mode = mode
        train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=args.preprocessed)
        train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
        val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

        run_dir = os.path.join(sweep_dir, mode)
        os.makedirs(run_dir, exist_ok=True)

        model = build_model(device)
        hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}
        model = train.main(
            device, train_ds, val_ds, model, hparams, run_dir,
            val_iter=args.val_iter, reg=args.weight, coef_norm=coef_norm,
        )

        hist_path = os.path.join(run_dir, f'loss_history_{args.nb_epochs}.json')
        with open(hist_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        summary = summarize(mode, history)
        summary['coef_norm'] = coef_norm_to_jsonable(coef_norm)
        summaries.append(summary)
        results.append({'norm_mode': mode, 'history': history, 'summary': summary})

        with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        # 释放显存，准备下一组
        del model, train_ds, val_ds, train_data, val_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(sweep_dir, 'comparison_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)

    plot_comparison(results, os.path.join(sweep_dir, 'norm_compare_l2.png'))

    print('\n===== Comparison Summary (lower L2 is better) =====')
    for s in summaries:
        print(
            f"  {s['norm_mode']:10s} | "
            f"final L2 press={s['final_val_l2_press']:.6f}  "
            f"velo={s['final_val_l2_velo']:.6f} | "
            f"best L2 press={s['best_val_l2_press']:.6f}@ep{s['best_val_l2_press_epoch']}  "
            f"velo={s['best_val_l2_velo']:.6f}@ep{s['best_val_l2_velo_epoch']}"
        )
    print(f'\nArtifacts: {sweep_dir}')


if __name__ == '__main__':
    main()

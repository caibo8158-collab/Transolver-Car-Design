import numpy as np
import time, json, os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from tqdm import tqdm
from dataset.dataset import denormalize_y, coef_norm_to_jsonable


def plot_loss_curves(history, save_path):
    """根据训练记录绘制 train_loss / val_loss 曲线。"""
    epochs = history['epoch']
    train_losses = history['train_loss']
    val_epochs = history['val_epoch']
    val_losses = history['val_loss']

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='train_loss', color='#1f77b4', linewidth=1.5)
    if val_epochs:
        plt.plot(val_epochs, val_losses, label='val_loss', color='#d62728',
                 marker='o', markersize=3, linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training / Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Loss curve saved to: {save_path}')


def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, reg=1):
    model.train()

    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    for cfd_data, geom in train_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        optimizer.zero_grad()
        out = model((cfd_data, geom))
        targets = cfd_data.y

        loss_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()
        total_loss = loss_velo + reg * loss_press

        total_loss.backward()

        optimizer.step()
        scheduler.step()

        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())

    return np.mean(losses_press), np.mean(losses_velo)


@torch.no_grad()
def test(device, model, test_loader, coef_norm=None):
    """
    返回标准化空间的 (loss_press, loss_velo)。
    若传入 coef_norm，额外在物理量纲下统计相对 L2（跨归一化方式可公平对比）。
    """
    model.eval()

    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    l2_press_list = []
    l2_velo_list = []
    for cfd_data, geom in test_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        out = model((cfd_data, geom))
        targets = cfd_data.y

        loss_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()

        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())

        if coef_norm is not None:
            pred = denormalize_y(out, cfd_data.surf, coef_norm)
            gt = denormalize_y(targets, cfd_data.surf, coef_norm)
            p_pred, p_gt = pred[cfd_data.surf, -1], gt[cfd_data.surf, -1]
            v_pred, v_gt = pred[~cfd_data.surf, :-1], gt[~cfd_data.surf, :-1]
            l2_press_list.append((torch.norm(p_pred - p_gt) / (torch.norm(p_gt) + 1e-8)).item())
            l2_velo_list.append((torch.norm(v_pred - v_gt) / (torch.norm(v_gt) + 1e-8)).item())

    press_m = float(np.mean(losses_press))
    velo_m = float(np.mean(losses_velo))
    if coef_norm is not None:
        return press_m, velo_m, float(np.mean(l2_press_list)), float(np.mean(l2_velo_list))
    return press_m, velo_m


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, val_dataset, Net, hparams, path, reg=1, val_iter=1, coef_norm=[]):
    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
        final_div_factor=1000.,
    )
    start = time.time()

    train_loss, val_loss = 1e5, 1e5
    history = {
        'epoch': [],
        'train_loss': [],
        'val_epoch': [],
        'val_loss': [],
        'val_l2_press': [],
        'val_l2_velo': [],
    }
    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
        loss_press, loss_velo = train(device, model, train_loader, optimizer, lr_scheduler, reg=reg)
        train_loss = loss_velo + reg * loss_press
        del (train_loader)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(float(train_loss))

        if val_iter is not None and (epoch == hparams['nb_epochs'] - 1 or epoch % val_iter == 0):
            val_loader = DataLoader(val_dataset, batch_size=1)

            test_out = test(device, model, val_loader, coef_norm=coef_norm if coef_norm else None)
            if len(test_out) == 4:
                loss_press, loss_velo, l2_press, l2_velo = test_out
                history['val_l2_press'].append(float(l2_press))
                history['val_l2_velo'].append(float(l2_velo))
            else:
                loss_press, loss_velo = test_out
                l2_press = l2_velo = None
            val_loss = loss_velo + reg * loss_press
            del (val_loader)

            history['val_epoch'].append(epoch + 1)
            history['val_loss'].append(float(val_loss))
            if l2_press is not None:
                pbar_train.set_postfix(train_loss=train_loss, val_loss=val_loss,
                                       l2_p=l2_press, l2_v=l2_velo)
            else:
                pbar_train.set_postfix(train_loss=train_loss, val_loss=val_loss)
        else:
            pbar_train.set_postfix(train_loss=train_loss)

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    torch.save(model, path + os.sep + f'model_{hparams["nb_epochs"]}.pth')

    history_path = path + os.sep + f'loss_history_{hparams["nb_epochs"]}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Loss history saved to: {history_path}')

    curve_path = path + os.sep + f'loss_curve_{hparams["nb_epochs"]}.png'
    plot_loss_curves(history, curve_path)

    if val_iter is not None:
        with open(path + os.sep + f'log_{hparams["nb_epochs"]}.json', 'a') as f:
            json.dump(
                {
                    'nb_parameters': params_model,
                    'time_elapsed': time_elapsed,
                    'hparams': hparams,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'final_val_l2_press': history['val_l2_press'][-1] if history['val_l2_press'] else None,
                    'final_val_l2_velo': history['val_l2_velo'][-1] if history['val_l2_velo'] else None,
                    'coef_norm': coef_norm_to_jsonable(coef_norm),
                }, f, indent=12, cls=NumpyEncoder
            )

    return model

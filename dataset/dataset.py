import torch
import vtk
import os
import itertools
import random
import numpy as np
from torch_geometric import nn as nng
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import k_hop_subgraph, subgraph
from vtk.util.numpy_support import vtk_to_numpy


# ========================= VTK / 几何工具 =========================

def load_unstructured_grid_data(file_name):
    """读取 VTK 非结构网格文件（.vtk）"""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output


def unstructured_grid_data_to_poly_data(unstructured_grid_data):
    """非结构网格 → 表面 PolyData（用于法向等计算）"""
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter


def get_sdf(target, boundary):
    """
    计算有符号距离场近似：每个 target 点到 boundary 最近点的距离与方向。
    返回: dists (N,), dirs (N, 3)
    """
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-8)
    return dists.reshape(-1), dirs


def get_normal(unstructured_grid_data):
    """从表面网格计算点法向，并做归一化"""
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(unstructured_grid_data)
    # visualize_poly_data(poly_data, surface_filter)
    # poly_data.GetPointData().SetScalars(None)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    # normal_filter.SetSplitting(0)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()
    '''
    normal_filter.SetComputeCellNormals(0)
    normal_filter.SetComputePointNormals(1)
    normal_filter.Update()
    #visualize_poly_data(poly_data, surface_filter, normal_filter)
    poly_data.GetPointData().SetNormals(normal_filter.GetOutput().GetPointData().GetNormals())
    p2c = vtk.vtkPointDataToCellData()
    p2c.ProcessAllArraysOn()
    p2c.SetInputData(poly_data)
    p2c.Update()
    unstructured_grid_data.GetCellData().SetNormals(p2c.GetOutput().GetCellData().GetNormals())
    #visualize_poly_data(poly_data, surface_filter, p2c)
    '''

    # 单元法向 → 点法向
    unstructured_grid_data.GetCellData().SetNormals(normal_filter.GetOutput().GetCellData().GetNormals())
    c2p = vtk.vtkCellDataToPointData()
    # c2p.ProcessAllArraysOn()
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()
    unstructured_grid_data = c2p.GetOutput()
    # return unstructured_grid_data
    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.double)
    # print(np.max(np.max(np.abs(normal), axis=1)), np.min(np.max(np.abs(normal), axis=1)))
    normal /= (np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8)
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
    # 出现 NaN 时递归重算一次
    if np.isnan(normal).sum() > 0:
        print(np.isnan(normal).sum())
        print("recalculate")
        return get_normal(unstructured_grid_data)  # re-calculate
    # print(normal)
    return normal


def visualize_poly_data(poly_data, surface_filter, normal_filter=None):
    """调试用：可视化表面网格 / 法向箭头（训练时一般不调用）"""
    if normal_filter is not None:
        mask = vtk.vtkMaskPoints()
        mask.SetInputData(normal_filter.GetOutput())
        # mask.RandomModeOn()
        mask.Update()
        arrow = vtk.vtkArrowSource()
        arrow.Update()
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(mask.GetOutput())
        glyph.SetSourceData(arrow.GetOutput())
        glyph.SetVectorModeToUseNormal()
        glyph.SetScaleFactor(0.1)
        glyph.Update()
        norm_mapper = vtk.vtkPolyDataMapper()
        norm_mapper.SetInputData(normal_filter.GetOutput())
        glyph_mapper = vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputData(glyph.GetOutput())
        norm_actor = vtk.vtkActor()
        norm_actor.SetMapper(norm_mapper)
        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        glyph_actor.GetProperty().SetColor(1, 0, 0)
        norm_render = vtk.vtkRenderer()
        norm_render.AddActor(norm_actor)
        norm_render.SetBackground(0, 1, 0)
        glyph_render = vtk.vtkRenderer()
        glyph_render.AddActor(glyph_actor)
        glyph_render.AddActor(norm_actor)
        glyph_render.SetBackground(0, 0, 1)

    scalar_range = poly_data.GetScalarRange()

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(surface_filter.GetOutputPort())
    mapper.SetScalarRange(scalar_range)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Set background to white

    renderer_window = vtk.vtkRenderWindow()
    renderer_window.AddRenderer(renderer)
    if normal_filter is not None:
        renderer_window.AddRenderer(norm_render)
        renderer_window.AddRenderer(glyph_render)
    renderer_window.Render()

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderer_window)
    interactor.Initialize()
    interactor.Start()


# ========================= 数据预处理主函数 =========================

def get_datalist(root, samples, norm=False, coef_norm=None, savedir=None, preprocessed=False):
    """
    将样本列表转为 PyG Data 列表（可选标准化，并缓存到 savedir）。

    参数:
        root: 原始数据根目录（含 param*/样本/vtk）
        samples: 样本相对路径列表，如 ['param0/xxx', ...]
        norm: 是否标准化
        coef_norm: 已有归一化系数 (mean_in, std_in, mean_out, std_out)；验证集传入训练集系数
        savedir: 预处理缓存目录
        preprocessed: True 则直接读 npy；False 则从 VTK 现场处理

    每个点特征:
        x (init): [x,y,z, sdf, nx,ny,nz] 共 7 维
        y (target): [vx,vy,vz, p] 共 4 维
        surf: 是否为表面点
    """
    dataset = []
    mean_in, mean_out = 0, 0
    std_in, std_out = 0, 0
    for k, s in enumerate(samples):
        # ----- 已预处理：直接读 npy -----
        if preprocessed and savedir is not None:
            save_path = os.path.join(savedir, s)
            if not os.path.exists(save_path):
                continue
            init = np.load(os.path.join(save_path, 'x.npy'))
            target = np.load(os.path.join(save_path, 'y.npy'))
            pos = np.load(os.path.join(save_path, 'pos.npy'))
            surf = np.load(os.path.join(save_path, 'surf.npy'))
            edge_index = np.load(os.path.join(save_path, 'edge_index.npy'))
        else:
            # ----- 未预处理：从 VTK 读取并构造特征 -----
            file_name_press = os.path.join(root, os.path.join(s, 'quadpress_smpl.vtk'))  # 表面压力
            file_name_velo = os.path.join(root, os.path.join(s, 'hexvelo_smpl.vtk'))    # 体积速度

            if not os.path.exists(file_name_press) or not os.path.exists(file_name_velo):
                continue

            unstructured_grid_data_press = load_unstructured_grid_data(file_name_press)
            unstructured_grid_data_velo = load_unstructured_grid_data(file_name_velo)#读取两类数据

            # 物理量与坐标
            velo = vtk_to_numpy(unstructured_grid_data_velo.GetPointData().GetVectors())#抽出体积网络速度数据
            press = vtk_to_numpy(unstructured_grid_data_press.GetPointData().GetScalars())#抽出表面压力数据
            points_velo = vtk_to_numpy(unstructured_grid_data_velo.GetPoints().GetData())#抽出体积网络坐标数据
            points_press = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())#抽出表面坐标数据

            # 网格边：表面四边形 cell_size=4，体积六面体 cell_size=8
            edges_press = get_edges(unstructured_grid_data_press, points_press, cell_size=4)
            edges_velo = get_edges(unstructured_grid_data_velo, points_velo, cell_size=8)

            # 体积点相对车身表面的距离/方向；表面点 sdf=0，法向由网格算
            sdf_velo, normal_velo = get_sdf(points_velo, points_press)
            sdf_press = np.zeros(points_press.shape[0])
            normal_press = get_normal(unstructured_grid_data_press)

            # 分离：外部流场点 vs 车身表面点
            surface = {tuple(p) for p in points_press}  # 表面点坐标集合，用于判断重合
            exterior_indices = [i for i, p in enumerate(points_velo) if tuple(p) not in surface]  # 纯外部点下标
            velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}  # 坐标 → 速度，供表面点对齐

            # _ext：外部流场点；_surf：车身表面点
            pos_ext = points_velo[exterior_indices]   # 外部点坐标 (x,y,z)
            pos_surf = points_press                    # 表面点坐标
            sdf_ext = sdf_velo[exterior_indices]      # 外部点到车身距离
            sdf_surf = sdf_press                      # 表面点距离，全 0
            normal_ext = normal_velo[exterior_indices]  # 外部点：离最近表面点的方向
            normal_surf = normal_press                  # 表面点：几何法向
            velo_ext = velo[exterior_indices]           # 外部点速度 (vx,vy,vz)
            # 表面点速度：体积网格有同坐标则取，否则填 0
            velo_surf = np.array([velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3) for p in pos_surf])
            press_ext = np.zeros([len(exterior_indices), 1])  # 体积点不监督压力，填 0
            press_surf = press                          # 表面压力真值（要监督）

            # 输入 / 标签拼接
            init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
            init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
            target_ext = np.c_[velo_ext, press_ext]
            target_surf = np.c_[velo_surf, press_surf]

            # 行方向拼接：先外部点，再表面点，合成一个完整样本
            surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])  # 0=外部，1=表面
            pos = np.concatenate([pos_ext, pos_surf])          # 全部点坐标
            init = np.concatenate([init_ext, init_surf])      # 全部点输入 x，7 维
            target = np.concatenate([target_ext, target_surf])  # 全部点标签 y，4 维
            # 用合并后的 pos 把表面边+体积边映射为统一节点编号
            edge_index = get_edge_index(pos, edges_press, edges_velo)

            # 缓存到 savedir，供下次 --preprocessed 1 直接读取
            if savedir is not None:
                save_path = os.path.join(savedir, s)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, 'x.npy'), init)
                np.save(os.path.join(save_path, 'y.npy'), target)
                np.save(os.path.join(save_path, 'pos.npy'), pos)
                np.save(os.path.join(save_path, 'surf.npy'), surf)
                np.save(os.path.join(save_path, 'edge_index.npy'), edge_index)

        # 转为 tensor，封装为 PyG Data
        surf = torch.tensor(surf)
        pos = torch.tensor(pos)
        x = torch.tensor(init)
        y = torch.tensor(target)
        edge_index = torch.tensor(edge_index)

        # 训练集：在线累积输入/输出均值（后续再算 std）
        if norm and coef_norm is None:
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis=0)
                mean_out = target.mean(axis=0)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis=0) - init.shape[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0) - init.shape[0] * mean_out) / new_length
                old_length = new_length
        data = Data(pos=pos, x=x, y=y, surf=surf.bool(), edge_index=edge_index)
        # data = Data(pos=pos, x=x, y=y, surf=surf.bool())
        dataset.append(data)

    # 训练集：算标准差并标准化，返回 (dataset, coef_norm)
    if norm and coef_norm is None:
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0) / old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (((data.x.numpy() - mean_in) ** 2).sum(axis=0) - data.x.numpy().shape[
                    0] * std_in) / new_length
                std_out += (((data.y.numpy() - mean_out) ** 2).sum(axis=0) - data.x.numpy().shape[
                    0] * std_out) / new_length
                old_length = new_length

        std_in = np.sqrt(std_in)
        std_out = np.sqrt(std_out)

        for data in dataset:
            data.x = ((data.x - mean_in) / (std_in + 1e-8)).float()
            data.y = ((data.y - mean_out) / (std_out + 1e-8)).float()

        coef_norm = (mean_in, std_in, mean_out, std_out)
        dataset = (dataset, coef_norm)

    # 验证集：使用训练集的 coef_norm 做同样标准化
    elif coef_norm is not None:
        for data in dataset:
            data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)).float()
            data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)).float()

    return dataset


# ========================= 构图相关 =========================

def get_edges(unstructured_grid_data, points, cell_size=4):
    """从单元拓扑提取边，边端点用坐标 tuple 表示（便于后续与 pos 对齐）"""
    edge_indeces = set()
    cells = vtk_to_numpy(unstructured_grid_data.GetCells().GetData()).reshape(-1, cell_size + 1)
    for i in range(len(cells)):
        for j, k in itertools.product(range(1, cell_size + 1), repeat=2):
            edge_indeces.add((cells[i][j], cells[i][k]))
            edge_indeces.add((cells[i][k], cells[i][j]))
    edges = [[], []]
    for u, v in edge_indeces:
        edges[0].append(tuple(points[u]))
        edges[1].append(tuple(points[v]))
    return edges


def get_edge_index(pos, edges_press, edges_velo):
    """把坐标形式的边映射为节点编号 edge_index，形状 [2, E]"""
    indices = {tuple(pos[i]): i for i in range(len(pos))}
    edges = set()
    for i in range(len(edges_press[0])):
        edges.add((indices[edges_press[0][i]], indices[edges_press[1][i]]))
    for i in range(len(edges_velo[0])):
        edges.add((indices[edges_velo[0][i]], indices[edges_velo[1][i]]))
    edge_index = np.array(list(edges)).T
    return edge_index


def get_induced_graph(data, idx, num_hops):
    """取节点 idx 的 k-hop 子图（本任务主流程未使用）"""
    subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx=idx, num_hops=num_hops, edge_index=data.edge_index,
                                                  relabel_nodes=True)
    return Data(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)


def pc_normalize(pc):
    """点云中心化并缩放到单位球内"""
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def get_shape(data, max_n_point=8192, normalize=True, use_height=False):
    """从表面点采样几何点云（作为模型的 geom 输入）"""
    surf_indices = torch.where(data.surf)[0].tolist()

    # 表面点过多时随机下采样
    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))

    shape_pc = data.pos[surf_indices].clone()

    if normalize:
        shape_pc = pc_normalize(shape_pc)

    # 可选：附加高度特征
    if use_height:
        gravity_dim = 1
        height_array = shape_pc[:, gravity_dim:gravity_dim + 1] - shape_pc[:, gravity_dim:gravity_dim + 1].min()
        shape_pc = torch.cat((shape_pc, height_array), axis=1)

    return shape_pc


def create_edge_index_radius(data, r, max_neighbors=32):
    """不用 CFD 网格时：按半径 r 建图"""
    data.edge_index = nng.radius_graph(x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors)
    # print(f'r = {r}, #edges = {data.edge_index.size(1)}')
    return data


# ========================= PyG Dataset 封装 =========================

class GraphDataset(Dataset):
    """训练/验证用数据集：返回 (cfd_data, shape_geom)"""

    def __init__(self, datalist, use_height=False, use_cfd_mesh=True, r=None):
        super().__init__()
        self.datalist = datalist
        self.use_height = use_height
        # use_cfd_mesh=False 时丢弃网格边，改用半径图
        if not use_cfd_mesh:
            assert r is not None
            for i in range(len(self.datalist)):
                self.datalist[i] = create_edge_index_radius(self.datalist[i], r)

    def len(self):
        return len(self.datalist)

    def get(self, idx):
        data = self.datalist[idx]
        shape = get_shape(data, use_height=self.use_height)
        return self.datalist[idx], shape


# ========================= 单样本调试入口 =========================

if __name__ == '__main__':
    import numpy as np

    file_name = '1a0bc9ab92c915167ae33d942430658c'

    root = '/Users/caibo/Desktop/Transolver-main/mlcfd_data/training_data'
    save_path = '/Users/caibo/Desktop/Transolver-main/mlcfd_data/preprocessed_data/param0/' + file_name
    file_name_press = 'param0/' + file_name + '/quadpress_smpl.vtk'
    file_name_velo = 'param0/' + file_name + '/hexvelo_smpl.vtk'
    file_name_press = os.path.join(root, file_name_press)
    file_name_velo = os.path.join(root, file_name_velo)
    unstructured_grid_data_press = load_unstructured_grid_data(file_name_press)
    unstructured_grid_data_velo = load_unstructured_grid_data(file_name_velo)

    velo = vtk_to_numpy(unstructured_grid_data_velo.GetPointData().GetVectors())
    press = vtk_to_numpy(unstructured_grid_data_press.GetPointData().GetScalars())
    points_velo = vtk_to_numpy(unstructured_grid_data_velo.GetPoints().GetData())
    points_press = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())

    edges_press = get_edges(unstructured_grid_data_press, points_press, cell_size=4)
    edges_velo = get_edges(unstructured_grid_data_velo, points_velo, cell_size=8)

    sdf_velo, normal_velo = get_sdf(points_velo, points_press)
    sdf_press = np.zeros(points_press.shape[0])
    normal_press = get_normal(unstructured_grid_data_press)

    surface = {tuple(p) for p in points_press}
    exterior_indices = [i for i, p in enumerate(points_velo) if tuple(p) not in surface]
    velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}

    pos_ext = points_velo[exterior_indices]
    pos_surf = points_press
    sdf_ext = sdf_velo[exterior_indices]
    sdf_surf = sdf_press
    normal_ext = normal_velo[exterior_indices]
    normal_surf = normal_press
    velo_ext = velo[exterior_indices]
    velo_surf = np.array([velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3) for p in pos_surf])
    press_ext = np.zeros([len(exterior_indices), 1])
    press_surf = press

    init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
    init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
    target_ext = np.c_[velo_ext, press_ext]
    target_surf = np.c_[velo_surf, press_surf]

    surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])
    pos = np.concatenate([pos_ext, pos_surf])
    init = np.concatenate([init_ext, init_surf])
    target = np.concatenate([target_ext, target_surf])

    edge_index = get_edge_index(pos, edges_press, edges_velo)

    data = Data(pos=torch.tensor(pos), edge_index=torch.tensor(edge_index))
    data = create_edge_index_radius(data, r=0.2)
    x, y = data.edge_index
    import torch_geometric

    print(max(torch_geometric.utils.degree(x)), max(torch_geometric.utils.degree(y)))

    print(points_velo.shape, points_press.shape)
    print(surf.shape, pos.shape, init.shape, target.shape, edge_index.shape)

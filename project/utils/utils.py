import os
from argparse import ArgumentParser

import dgl
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from SE(3)-Transformer (https://github.com/FabianFuchsML/se3-transformer-public/):
# -------------------------------------------------------------------------------------------------------------------------------------

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from GraphTransformer (https://github.com/graphdeeplearning/graphtransformer/):
# -------------------------------------------------------------------------------------------------------------------------------------
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        """Compute the dot product between source nodes' and destination nodes' representations."""
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling(field, scale_constant, clip_constant):
    def func(edges):
        """Scale edge representation value using a constant divisor."""
        return {field: ((edges.data[field]) / scale_constant).clamp(-clip_constant, clip_constant)}

    return func


def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        """Improve implicit attention scores with explicit edge features, if available."""
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


def out_edge_features(edge_feat):
    def func(edges):
        """Copy edge features to be passed to FFN_e."""
        return {'e_out': edges.data[edge_feat]}

    return func


def exp(field, clip_constant):
    def func(edges):
        """Clamp edge representations for softmax numerical stability."""
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-clip_constant, clip_constant))}

    return func


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for EGNN-DGL (https://github.com/amorehead/EGNN-DGL):
# -------------------------------------------------------------------------------------------------------------------------------------
def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)


def glorot_orthogonal(tensor, scale):
    """Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def calculate_and_store_dists_in_graph(graph: dgl.DGLGraph):
    """Derive all node-node distance features from a given batch of DGLGraphs."""
    graphs = dgl.unbatch(graph)
    for graph in graphs:
        graph.edata['c'] = graph.ndata['x'][graph.edges()[1]] - graph.ndata['x'][graph.edges()[0]]
        graph.edata['r'] = torch.sum(graph.edata['c'] ** 2, 1).reshape(-1, 1)
    graph = dgl.batch(graphs)
    return graph


def get_graph(src, dst, pos, node_feature, edge_feature, dtype, undirected=True, num_nodes=None):
    """Construct a single DGLGraph given source and destination node IDs, coordinates, and node and edge features."""
    # src, dst : indices for vertices of source and destination, torch.Tensor
    # pos: x, y, z coordinates of all vertices with respect to the indices, torch.Tensor
    # node_feature: node feature of shape [num_nodes, node_feature_size], torch.Tensor
    # edge_feature: edge feature of shape [num_nodes, edge_feature_size], torch.Tensor
    if num_nodes:
        G = dgl.graph((src, dst), num_nodes=num_nodes)
    else:
        G = dgl.graph((src, dst))
    if undirected:
        G = dgl.to_bidirected(G)
    # Add node features to graph
    G.ndata['f'] = node_feature.type(dtype)
    G.ndata['x'] = pos.type(dtype)  # [num_nodes, 3]
    # Add edge features to graph
    G.edata['c'] = pos[dst] - pos[src]  # [num_nodes, 3]
    G.edata['f'] = edge_feature.type(dtype)  # [num_nodes, edge_feature_size]
    return G


def get_rgraph(num_nodes: int, num_edges: int, node_feature_size: int,
               edge_feature_size: int, dtype: torch.Type, test: bool):
    G = dgl.rand_graph(num_nodes, num_edges)
    if test:
        src = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        dst = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])
    else:
        src = G.edges()[0]
        dst = G.edges()[1]
    # Add node features to graph
    pos = torch.ones(num_nodes, 3) if test else torch.rand((num_nodes, 3))  # [num_nodes, 3]
    node_features = torch.ones(num_nodes, node_feature_size) if test else torch.rand((num_nodes, node_feature_size))
    # Add edge features to graph
    edge_features = torch.ones(num_edges, edge_feature_size) if test else torch.rand((num_edges, edge_feature_size))
    return get_graph(src, dst, pos, node_features, edge_features, dtype, False, num_nodes=num_nodes)


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from egnn-pytorch (https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/utils.py):
# -------------------------------------------------------------------------------------------------------------------------------------
def rot_z(gamma):
    return torch.tensor([
        [torch.cos(gamma), -torch.sin(gamma), 0],
        [torch.sin(gamma), torch.cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)


def rot_y(beta):
    return torch.tensor([
        [torch.cos(beta), 0, torch.sin(beta)],
        [0, 1, 0],
        [-torch.sin(beta), 0, torch.cos(beta)]
    ], dtype=beta.dtype)


def rotate(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def collect_args():
    """Collect all arguments required for training/testing."""
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # -----------------
    # Model arguments
    # -----------------
    parser.add_argument('--metric_to_track', type=str, default='val_mse', help='Scheduling and early stop')

    # -----------------
    # Logging arguments
    # -----------------
    parser.add_argument('--logger_name', type=str, default='TensorBoard', help='Which logger to use for experiments')
    parser.add_argument('--experiment_name', type=str, default=None, help='Logger experiment name')
    parser.add_argument('--project_name', type=str, default='EGNN-DGL', help='Logger project name')
    parser.add_argument('--entity', type=str, default='DGL', help='Logger entity (i.e. team) name')
    parser.add_argument('--run_id', type=str, default='', help='Logger run ID')
    parser.add_argument('--offline', action='store_true', dest='offline', help='Whether to log locally or remotely')
    parser.add_argument('--online', action='store_false', dest='offline', help='Whether to log locally or remotely')
    parser.add_argument('--tb_log_dir', type=str, default='tb_logs', help='Where to store TensorBoard log files')
    parser.set_defaults(offline=False)  # Default to using online logging mode

    # -----------------
    # Seed arguments
    # -----------------
    parser.add_argument('--seed', type=int, default=None, help='Seed for NumPy and PyTorch')

    # -----------------
    # Meta-arguments
    # -----------------
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples included in each data batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Decay rate of optimizer weight')
    parser.add_argument('--num_epochs', type=int, default=50, help='Maximum number of epochs to run for training')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout (forget) rate')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait until early stopping')

    # -----------------
    # Miscellaneous
    # -----------------
    parser.add_argument('--max_hours', type=int, default=1, help='Maximum number of hours to allot for training')
    parser.add_argument('--max_minutes', type=int, default=55, help='Maximum number of minutes to allot for training')
    parser.add_argument('--multi_gpu_backend', type=str, default='ddp', help='Multi-GPU backend for training')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (e.g. -1 = all available GPUs)')
    parser.add_argument('--auto_choose_gpus', action='store_true', dest='auto_choose_gpus', help='Auto-select GPUs')
    parser.add_argument('--num_compute_nodes', type=int, default=1, help='Number of compute nodes to use')
    parser.add_argument('--gpu_precision', type=int, default=32, help='Bit size used during training (e.g. 16-bit)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU threads for loading data')
    parser.add_argument('--profiler_method', type=str, default=None, help='PL profiler to use (e.g. simple)')
    parser.add_argument('--ckpt_dir', type=str, default=f'{os.path.join(os.getcwd(), "checkpoints")}',
                        help='Directory in which to save checkpoints')
    parser.add_argument('--ckpt_name', type=str, default='', help='Filename of best checkpoint')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum percentage of change required to'
                                                                       ' "metric_to_track" before early stopping'
                                                                       ' after surpassing patience')
    parser.add_argument('--accum_grad_batches', type=int, default=1, help='Norm over which to clip gradients')
    parser.add_argument('--grad_clip_val', type=float, default=0.5, help='Norm over which to clip gradients')
    parser.add_argument('--grad_clip_algo', type=str, default='norm', help='Algorithm with which to clip gradients')
    parser.add_argument('--stc_weight_avg', action='store_true', dest='stc_weight_avg', help='Smooth loss landscape')
    parser.add_argument('--find_lr', action='store_true', dest='find_lr', help='Find an optimal learning rate a priori')

    return parser


def process_args(args):
    """Process all arguments required for training/testing."""
    # ---------------------------------------
    # Seed fixing for random numbers
    # ---------------------------------------
    if not args.seed:
        args.seed = 42  # np.random.randint(100000)
    print(f'Seeding everything with random seed {args.seed}')
    pl.seed_everything(args.seed)
    dgl.seed(args.seed)

    return args


def construct_pl_logger(args):
    """Return a specific Logger instance requested by the user."""
    if args.logger_name.lower() == 'wandb':
        return construct_wandb_pl_logger(args)
    else:  # Default to using TensorBoard
        return construct_tensorboard_pl_logger(args)


def construct_wandb_pl_logger(args):
    """Return an instance of WandbLogger with corresponding project and name strings."""
    return WandbLogger(name=args.experiment_name,
                       offline=args.offline,
                       project=args.project_name,
                       log_model=True,
                       entity=args.entity)


def construct_tensorboard_pl_logger(args):
    """Return an instance of TensorBoardLogger with corresponding project and experiment name strings."""
    return TensorBoardLogger(save_dir=args.tb_log_dir,
                             name=args.experiment_name)

import torch
import torch.nn as nn

from project.datasets.RG.rg_dgl_dataset import get_rgraph
from project.utils.modules import DGLEnGraphConv
from project.utils.utils import rotate


def test_dgl_egnn():
    # Declare hyperparameters for testing
    num_nodes = 4
    num_edges = 12
    num_node_input_feats = 1
    num_gnn_hidden_channels = 32
    num_edge_input_feats = 1
    num_test_layers = 4
    activ_fn = nn.GELU() if num_test_layers > 16 else nn.SiLU()
    tanh = num_test_layers > 16
    test_against_orig_egnn = False

    # Initialize EGNN embedding layers
    using_node_embeddings = num_node_input_feats != num_gnn_hidden_channels
    using_edge_embedding = num_edge_input_feats != num_gnn_hidden_channels
    node_in_embedding = nn.Linear(num_node_input_feats, num_gnn_hidden_channels) \
        if using_node_embeddings \
        else nn.Identity()
    node_out_embedding = nn.Linear(num_gnn_hidden_channels, num_node_input_feats) \
        if using_node_embeddings \
        else nn.Identity()
    edge_in_embedding = nn.Linear(num_edge_input_feats, num_gnn_hidden_channels) \
        if using_edge_embedding \
        else nn.Identity()

    # Initialize DGL EGNN layers for testing
    layer = DGLEnGraphConv(
        num_input_feats=num_gnn_hidden_channels,
        num_hidden_feats=num_gnn_hidden_channels,
        num_output_feats=num_gnn_hidden_channels,
        num_edge_input_feats=num_edge_input_feats,
        activ_fn=activ_fn,
        residual=True,
        simple_attention=False,
        adv_attention=False,
        num_attention_heads=4,
        attention_use_bias=False,
        norm_to_apply='batch',
        normalize_coord_diff=False,
        tanh=tanh,
        dropout=0.1,
        coords_aggr='mean',
        update_feats=True,
        update_coords=True,
    )
    layers = nn.ModuleList([layer])
    if test_against_orig_egnn:
        layers = nn.ModuleList(layer for _ in range(num_test_layers))

    # Create random transformation tensors
    R = rotate(*torch.rand(3))
    T = torch.randn(1, 3)

    # Generate random graph
    rgraph = get_rgraph(
        num_nodes=num_nodes,
        num_edges=num_edges,
        node_feature_size=num_node_input_feats,
        edge_feature_size=num_edge_input_feats,
        dtype=torch.FloatTensor,
        test=test_against_orig_egnn
    )

    # Assemble node features for propagation
    node_feats = rgraph.ndata['f']
    node_coords = rgraph.ndata['x']
    edge_feats = rgraph.edata['f']

    # Cache first two nodes' features
    node1 = node_feats[0, :]
    node2 = node_feats[1, :]

    # Switch first and second nodes' positions
    node_feats_permuted_row_wise = node_feats.clone().detach()
    node_feats_permuted_row_wise[0, :] = node2
    node_feats_permuted_row_wise[1, :] = node1

    # Embed input node features
    if test_against_orig_egnn:
        node_feats[0][0] = 0.5  # Arbitrarily substitute first node's first feature to see effect of message passing
    # Embed input features a priori
    if using_node_embeddings:
        node_feats = node_in_embedding(node_feats)
        node_feats_permuted_row_wise = node_in_embedding(node_feats_permuted_row_wise)
    if using_edge_embedding:
        edge_feats = edge_in_embedding(edge_feats)

    # Convolve over graph nodes and edges
    node_feats1, node_coords1, edge_feats1 = layer(
        rgraph, node_feats, node_coords @ R + T, edge_feats
    )
    node_feats2, node_coords2, edge_feats2 = layer(
        rgraph, node_feats, node_coords, edge_feats
    )
    node_feats3, node_coords3, edge_feats3 = layer(
        rgraph, node_feats_permuted_row_wise, node_coords, edge_feats
    )

    # Project learned node features' dimensionality a posteriori
    node_feats1 = node_out_embedding(node_feats1)
    node_feats2 = node_out_embedding(node_feats2)
    node_feats3 = node_out_embedding(node_feats3)

    assert torch.allclose(node_feats1, node_feats2, atol=1e-6), 'Type 0 features are invariant'
    assert torch.allclose(node_coords1, (node_coords2 @ R + T), atol=1e-4), 'Type 1 features are equivariant'
    assert not torch.allclose(node_feats1, node_feats3, atol=1e-6), 'Layer must be equivariant to node permutations'

    # Run specific test case against original EGNN layer
    if test_against_orig_egnn:
        test_node_feats = node_feats.clone().detach()
        test_node_coords = node_coords.clone().detach()
        test_edge_feats = edge_feats.clone().detach()
        for test_layer in layers:
            test_node_feats, test_node_coords, test_edge_feats = test_layer(
                rgraph, test_node_feats, test_node_coords, test_edge_feats
            )
        test_node_feats = node_out_embedding(test_node_feats)
        print(test_node_feats, test_node_coords)


if __name__ == '__main__':
    test_dgl_egnn()

from argparse import ArgumentParser
from typing import List

import dgl
import dgl.function as fn
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from project.utils.utils import glorot_orthogonal, calculate_and_store_dists_in_graph, src_dot_dst, scaling, \
    imp_exp_attn, out_edge_features, exp


# ------------------
# PyTorch Modules
# ------------------

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from 'A Generalization of Transformer Networks to Graphs' (https://github.com/graphdeeplearning/graphtransformer):
# -------------------------------------------------------------------------------------------------------------------------------------
class MultiHeadAttentionLayer(nn.Module):
    """Compute attention scores with a DGLGraph's node and edge features."""

    def __init__(self, num_input_feats: int, num_output_feats: int, num_heads: int, using_bias: bool):
        super().__init__()

        # Declare shared variables
        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias

        # Define node features' query, key, and value tensors, and define edge features' projection tensors
        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        if self.using_bias:
            glorot_orthogonal(self.Q.weight, scale=scale)
            self.Q.bias.data.fill_(0)

            glorot_orthogonal(self.K.weight, scale=scale)
            self.K.bias.data.fill_(0)

            glorot_orthogonal(self.V.weight, scale=scale)
            self.V.bias.data.fill_(0)

            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
            self.edge_feats_projection.bias.data.fill_(0)
        else:
            glorot_orthogonal(self.Q.weight, scale=scale)
            glorot_orthogonal(self.K.weight, scale=scale)
            glorot_orthogonal(self.V.weight, scale=scale)
            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)

    def propagate_attention(self, graph: dgl.DGLGraph):
        # Compute attention scores
        graph.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        # Scale and clip attention scores
        graph.apply_edges(scaling('score', np.sqrt(self.num_output_feats), 5.0))

        # Use available edge features to modify the attention scores
        graph.apply_edges(imp_exp_attn('score', 'proj_e'))

        # Copy edge features as e_out to be passed to edge_feats_MLP
        graph.apply_edges(out_edge_features('score'))

        # Apply softmax to attention scores, followed by clipping
        graph.apply_edges(exp('score', 5.0))

        # Send weighted values to target nodes
        e_ids = graph.edges()
        graph.send_and_recv(e_ids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        graph.send_and_recv(e_ids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, graph: dgl.DGLGraph, node_feats: torch.Tensor, edge_feats: torch.Tensor):
        with graph.local_scope():
            node_feats_q = self.Q(node_feats)
            node_feats_k = self.K(node_feats)
            node_feats_v = self.V(node_feats)
            edge_feats_projection = self.edge_feats_projection(edge_feats)

            # Reshape tensors into [num_nodes, num_heads, feat_dim] to get projections for multi-head attention
            graph.ndata['Q_h'] = node_feats_q.view(-1, self.num_heads, self.num_output_feats)
            graph.ndata['K_h'] = node_feats_k.view(-1, self.num_heads, self.num_output_feats)
            graph.ndata['V_h'] = node_feats_v.view(-1, self.num_heads, self.num_output_feats)
            graph.edata['proj_e'] = edge_feats_projection.view(-1, self.num_heads, self.num_output_feats)

            # Disperse attention information
            self.propagate_attention(graph)

            # Compute final node and edge representations after multi-head attention
            h_out = graph.ndata['wV'] / (graph.ndata['z'] + torch.full_like(graph.ndata['z'], 1e-6))  # Add eps to all
            e_out = graph.edata['e_out']

        # Return attention-updated node and edge representations
        return h_out, e_out


# ------------------
# DGL Modules
# ------------------

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for EGNN-DGL (https://github.com/amorehead/EGNN-DGL):
# -------------------------------------------------------------------------------------------------------------------------------------
class DGLEnGraphConv(nn.Module):
    """An E(n)-equivariant graph neural network layer as a DGL module.

    DGLEnGraphConv stands for a Graph Convolution E(n)-equivariant layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph
    conv layer in a GCN.
    """

    def __init__(
            self,
            num_input_feats: int,
            num_hidden_feats: int,
            num_output_feats: int,
            num_edge_input_feats: int,
            activ_fn=nn.SiLU(),
            residual=True,
            simple_attention=True,
            adv_attention=False,
            num_attention_heads=4,
            attention_use_bias=False,
            norm_to_apply='batch',
            normalize_coord_diff=False,
            tanh=False,
            coords_aggr='mean',
            update_coords=True,
            update_feats=True,
            self_loops=False,
            **kwargs
    ):
        """E(n)-equivariant Graph Conv Layer

        Parameters
        ----------
        num_input_feats : int
            Input feature size.
        num_hidden_feats : int
            Hidden feature size.
        num_output_feats : int
            Output feature size.
        activ_fn : Module
            Activation function to apply in MLPs.
        residual : bool
            Whether to use a residual update strategy for node features.
        simple_attention: int
            Whether to apply a simple version of attention to each inter-node message.
        adv_attention : bool
            Whether to apply an advanced multi-head attention operation on each edge message.
        num_attention_heads : int
            How many attention heads to apply to the input node features in parallel.
        attention_use_bias : bool
            Whether to include a bias term in the attention mechanism.
        norm_to_apply : str
            Which normalization scheme to apply to node and edge representations (i.e. 'layer' or 'batch').
        normalize_coord_diff : bool
            Whether to normalize the difference between node coordinates before applying MLPs.
        tanh : bool
            Whether to use a hyperbolic tangent function while applying the coordinates MLP.
        coords_aggr : str
            How to update coordinates (i.e. 'mean', 'sum').
        update_coords : bool
            Whether to update coordinates in an equivariant manner.
        update_feats : bool
            Whether to update node features in an invariant manner.
        self_loops : bool
            Whether the input graphs contain self-loops.
        """
        assert update_feats or update_coords, 'You must update either features, coordinates, or both.'
        super().__init__()

        # Initialize model parameters
        self.num_input_feats = num_input_feats
        self.num_hidden_feats = num_hidden_feats
        self.num_output_feats = num_output_feats
        self.num_edge_input_feats = num_edge_input_feats

        self.activ_fn = activ_fn
        self.residual = residual
        self.simple_attention = simple_attention
        self.adv_attention = adv_attention
        self.num_attention_heads = num_attention_heads
        self.attention_use_bias = attention_use_bias
        self.norm_to_apply = norm_to_apply
        self.normalize_coord_diff = normalize_coord_diff
        self.tanh = tanh
        self.coords_aggr = coords_aggr
        self.update_coords = update_coords
        self.update_feats = update_feats
        self.self_loops = self_loops

        # Pre-set parameters
        self.epsilon = 1e-8
        self.num_edge_coord_feats = 1

        # Define edge features multi-layer perceptron (MLP)
        self.edge_mlp_input_dim = (self.num_input_feats * 2) + self.num_edge_coord_feats + self.num_edge_input_feats
        self.edges_mlp = nn.Sequential(
            nn.Linear(self.edge_mlp_input_dim, self.num_hidden_feats),
            self.activ_fn,
            nn.Linear(self.num_hidden_feats, self.num_hidden_feats),
            self.activ_fn
        )

        # Define node features multi-layer perceptron (MLP)
        self.nodes_mlp_input_dim = self.num_hidden_feats + self.num_input_feats
        self.nodes_mlp = nn.Sequential(
            nn.Linear(self.nodes_mlp_input_dim, self.num_hidden_feats),
            self.activ_fn,
            nn.Linear(self.num_hidden_feats, self.num_output_feats)
        ) if self.update_feats else None

        # Initialize coordinates module a priori
        coords_module = nn.Linear(self.num_hidden_feats, 1, bias=False)
        torch.nn.init.xavier_uniform_(coords_module.weight, gain=0.001)

        # Define node coordinates multi-layer perceptron (MLP)
        if self.update_coords:
            self.coords_mlp = [nn.Linear(self.num_hidden_feats, self.num_hidden_feats)]
            self.coords_mlp.append(self.activ_fn)
            self.coords_mlp.append(coords_module)
            if self.tanh:
                self.coords_mlp.append(nn.Tanh())
            self.coords_mlp = nn.Sequential(*self.coords_mlp)

        # Define an optional simple attention (i.e. soft edge) multi-layer perceptron (MLP) for inter-node messages
        self.attn_mlp = nn.Sequential(
            nn.Linear(self.num_hidden_feats, 1),
            nn.Sigmoid()
        ) if self.simple_attention else None

        # Define an optional advanced multi-head attention (MHA) module for inter-node messages
        if self.adv_attention:
            self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

            self.mha_module = MultiHeadAttentionLayer(
                self.num_hidden_feats,
                self.num_output_feats // self.num_attention_heads,
                self.num_attention_heads,
                self.attention_use_bias
            )

            self.O_h = nn.Linear(self.num_output_feats, self.num_output_feats)
            self.O_e = nn.Linear(self.num_output_feats, self.num_output_feats)

            if self.apply_layer_norm:
                self.layer_norm1_h = nn.LayerNorm(self.num_output_feats)
                self.layer_norm1_e = nn.LayerNorm(self.num_output_feats)
            else:  # Otherwise, default to using batch normalization
                self.batch_norm1_h = nn.BatchNorm1d(self.num_output_feats)
                self.batch_norm1_e = nn.BatchNorm1d(self.num_output_feats)

            # FFN for h
            self.FFN_h_layer1 = nn.Linear(self.num_output_feats, self.num_output_feats * 2)
            self.FFN_h_layer2 = nn.Linear(self.num_output_feats * 2, self.num_output_feats)

            # FFN for e
            self.FFN_e_layer1 = nn.Linear(self.num_output_feats, self.num_output_feats * 2)
            self.FFN_e_layer2 = nn.Linear(self.num_output_feats * 2, self.num_output_feats)

            if self.apply_layer_norm:
                self.layer_norm2_h = nn.LayerNorm(self.num_output_feats)
                self.layer_norm2_e = nn.LayerNorm(self.num_output_feats)
            else:  # Otherwise, default to using batch normalization
                self.batch_norm2_h = nn.BatchNorm1d(self.num_output_feats)
                self.batch_norm2_e = nn.BatchNorm1d(self.num_output_feats)

    def apply_adv_attention(self, graph: dgl.DGLGraph, h: torch.Tensor, e: torch.Tensor):
        """Perform a forward pass using a graph multi-head attention module defined a priori."""
        h_in1 = h  # Cache node representations for first residual connection
        e_in1 = e  # Cache edge representations for first residual connection

        # Get multi-head attention output using previously-learned node representations and edge features
        h_attn_out, e_attn_out = self.mha_module(graph, h, e)

        h = h_attn_out.view(-1, self.num_output_feats)
        e = e_attn_out.view(-1, self.num_output_feats)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h  # Make first node residual connection
            e = e_in1 + e  # Make first edge residual connection

        if self.apply_layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)
        else:  # Otherwise, default to using batch normalization
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h  # Cache node representations for second residual connection
        e_in2 = e  # Cache edge representations for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h  # Make second node residual connection
            e = e_in2 + e  # Make second edge residual connection

        if self.apply_layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)
        else:  # Otherwise, default to using batch normalization
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        return h, e

    def message_func(self, edges: dgl.udf.EdgeBatch):
        """Compute the messages for an EdgeBatch of edges.
        This function is set up as a User Defined Function in DGL.

        Parameters
        ----------
        edges : EdgeBatch
            A batch of edges for which to compute messages
        """

        edge_mlp_input = torch.cat([edges.src['f'], edges.dst['f'], edges.data['r'], edges.data['f']], dim=1)
        m_ij = self.edges_mlp(edge_mlp_input)
        if self.simple_attention:  # Optionally apply an attention operation to messages
            # Apply a soft edge weight via a simple attention MLP
            attn_m_ij = self.attn_mlp(m_ij)
            m_ij = m_ij * attn_m_ij
        edges.data['m_ij'] = m_ij  # Preserve edge messages in the event that a reduce_func is called
        return {'m_ij': m_ij}

    @staticmethod
    def unsorted_segment_sum(data, segment_ids, num_segments):
        """Compute the sum of an unsorted segment of edges - Originally from https://github.com/vgsatorras/egnn."""
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)  # Initialize empty result tensor
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        return result

    @staticmethod
    def unsorted_segment_mean(data, segment_ids, num_segments):
        """Compute the mean of an unsorted segment of edges - Originally from https://github.com/vgsatorras/egnn."""
        result_shape = (num_segments, data.size(1))
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        count = data.new_full(result_shape, 0)
        result.scatter_add_(0, segment_ids, data)
        count.scatter_add_(0, segment_ids, torch.ones_like(data))
        return result / count.clamp(min=1)

    def update_node_coords(self, graph):
        """In an E(n) equivariant manner, update the node coordinates in the given graph."""
        if self.self_loops:
            edge_index_diffs = graph.edges()[0] - graph.edges()[1]
            selected_src_edge_indices = (edge_index_diffs != 0).nonzero(as_tuple=False).squeeze()
            coord_diffs = graph.edata['c'][selected_src_edge_indices]
            edge_messages = graph.edata['m_ij'][selected_src_edge_indices]
        else:
            selected_src_edge_indices = graph.edges()[0]  #
            coord_diffs = graph.edata['c']
            edge_messages = graph.edata['m_ij']

        x_trans = coord_diffs * self.coords_mlp(edge_messages)
        if self.coords_aggr == 'mean':  # Default to self.coords_aggr == 'mean'
            x_aggr = self.unsorted_segment_mean(
                x_trans, selected_src_edge_indices, num_segments=graph.ndata['x'].size(0)
            )
        elif self.coords_aggr == 'sum':
            x_aggr = self.unsorted_segment_sum(
                x_trans, selected_src_edge_indices, num_segments=graph.ndata['x'].size(0)
            )
        else:
            raise Exception(f'Invalid coords_aggr value supplied: {self.coords_aggr}')
        graph.ndata['x'] = graph.ndata['x'] + x_aggr
        return graph

    def update_node_feats(self, graph):
        """In an E(n) invariant manner, update the node features in the given graph."""
        graph.ndata['m_i'] = self.unsorted_segment_sum(
            graph.edata['m_ij'], graph.edges()[0], num_segments=graph.ndata['f'].size(0)
        )
        node_mlp_input = torch.cat([graph.ndata['f'], graph.ndata['m_i']], dim=1)
        node_mlp_out = self.nodes_mlp(node_mlp_input)
        if self.residual:
            graph.ndata['f'] = graph.ndata['f'] + node_mlp_out
        else:
            graph.ndata['f'] = node_mlp_out
        return graph

    def forward(
            self,
            graph: dgl.DGLGraph
    ):
        """Forward pass of the network

        Parameters
        ----------
        graph : DGLGraph
            DGL input graph
        """
        # Calculate (squared and unsquared) node-node distances in the given graph
        graph = calculate_and_store_dists_in_graph(graph)
        if self.normalize_coord_diff:  # Optionally normalize unsquared node-node distances
            graph.edata['c'] = graph.edata['c'] / (torch.sqrt(graph.edata['r']) + self.epsilon)

        # Apply an advanced multi-head attention module to the graph's node and edge features a priori
        if self.adv_attention:
            graph.ndata['f'], graph.edata['f'] = self.apply_adv_attention(
                graph,
                graph.ndata['f'],
                graph.edata['f']
            )

        # Craft all edge i->j (i.e. m_ij) messages
        graph.apply_edges(self.message_func)

        # Update node coordinates if requested
        if self.update_coords:
            graph = self.update_node_coords(graph)

        # Update node features if requested
        if self.update_feats:
            graph = self.update_node_feats(graph)

        return graph

    def __repr__(self):
        return f'DGLEnGraphConv(structure=' \
               f'h_in{self.num_input_feats}-h_hid{self.num_hidden_feats}-h_out{self.num_output_feats}' \
               f'-e_in{self.num_input_feats}-e_hid{self.num_hidden_feats}-e_out{self.num_output_feats})'


# ------------------
# Lightning Modules
# ------------------

class LitEGNN(pl.LightningModule):
    """A LightningModule for the DGL implementation of the Equivariant Graph Neural Network (EGNN)."""

    def __init__(self, num_node_input_feats: int, num_edge_input_feats: int, gnn_activ_fn=nn.SiLU(), num_gnn_layers=2,
                 num_gnn_hidden_channels=128, num_gnn_attention_heads=4, num_epochs=50, metric_to_track='val_mse',
                 weight_decay=1e-2, lr=1e-3):
        """Initialize all the parameters for a LitGINI module."""
        super().__init__()

        # Build the network
        self.num_node_input_feats = num_node_input_feats
        self.num_edge_input_feats = num_edge_input_feats
        self.gnn_activ_fn = gnn_activ_fn
        self.num_out_channels = 1

        # GNN module's keyword arguments provided via the command line
        self.num_gnn_layers = num_gnn_layers
        self.num_gnn_hidden_channels = num_gnn_hidden_channels
        self.num_gnn_attention_heads = num_gnn_attention_heads

        # Model hyperparameter keyword arguments provided via the command line
        self.num_epochs = num_epochs
        self.metric_to_track = metric_to_track
        self.weight_decay = weight_decay
        self.lr = lr

        # Set up GNN node and edge embedding layers (if requested)
        self.using_node_embedding = self.num_node_input_feats != self.num_gnn_hidden_channels
        self.node_in_embedding = nn.Linear(self.num_node_input_feats, self.num_gnn_hidden_channels, bias=False) \
            if self.using_node_embedding \
            else nn.Identity()

        # Assemble the layers of the network
        self.build_gnn_module(), self.build_fc_module()

        # Declare loss functions and metrics for training, validation, and testing
        self.loss_fn = nn.MSELoss()

        # Reset learnable parameters and log hyperparameters
        self.reset_parameters()
        self.save_hyperparameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.using_node_embedding:
            # Reinitialize node input embedding
            glorot_orthogonal(self.node_in_embedding.weight, scale=2.0)

    def build_gnn_module(self):
        """Define all layers for the chosen GNN module."""
        # Marshal all GNN layers, allowing the user to choose which kind of graph learning scheme they would like to use
        gnn_layers = [
            DGLEnGraphConv(
                num_input_feats=self.num_gnn_hidden_channels,
                num_hidden_feats=self.num_gnn_hidden_channels,
                num_output_feats=self.num_gnn_hidden_channels,
                num_edge_input_feats=self.num_edge_input_feats,
                activ_fn=self.gnn_activ_fn,
                residual=True,  # Whether to employ a residual connection during equivariant message-passing
                simple_attention=False,  # Whether to apply a simple gating-based attention mechanism on edge messages
                adv_attention=False,  # Whether to apply the Graph Transformer prior to equivariant message-passing
                num_attention_heads=self.num_gnn_attention_heads,
                attention_use_bias=False,
                norm_to_apply='batch',
                normalize_coord_diff=False,
                tanh=False,
                coords_aggr='mean',
                update_coords=True,
                update_feats=True
            )
            for _ in range(self.num_gnn_layers)
        ]
        self.gnn_module = nn.ModuleList(gnn_layers)

    def build_fc_module(self):
        self.fc_module = nn.Sequential(
            nn.Linear(self.num_gnn_hidden_channels, self.num_gnn_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_gnn_hidden_channels, self.num_out_channels)
        )

    # ---------------------
    # Training
    # ---------------------
    def gnn_forward(self, graph: dgl.DGLGraph):
        """Make a forward pass through a single GNN module."""
        # Embed input features a priori
        if self.using_node_embedding:
            graph.ndata['f'] = self.node_in_embedding(graph.ndata['f'].squeeze()).squeeze()
        # Forward propagate with each GNN layer
        for layer in self.gnn_module:
            # Cache the original batch number of nodes and edges
            batch_num_nodes, batch_num_edges = graph.batch_num_nodes(), graph.batch_num_edges()
            graph = layer(graph)
            # Retain the original batch number of nodes and edges
            graph.set_batch_num_nodes(batch_num_nodes), graph.set_batch_num_edges(batch_num_edges)
        # Return the updated graph batch
        graphs = dgl.unbatch(graph)
        logits_list = [batched_graph.ndata['f'] for batched_graph in graphs]
        return logits_list

    def fc_forward(self, logits_list: List[torch.Tensor]):
        """Forward propagate with the final fully-connected layer(s)."""
        fc_logits = []
        for logits in logits_list:
            fc_logits_summed = torch.sum(self.fc_module(logits), dim=0)
            fc_logits.append(fc_logits_summed)
        logits = torch.cat(fc_logits)
        return logits

    def shared_step(self, graph: dgl.DGLGraph):
        """Make a forward pass through the entire network."""
        # Learn from each input graph
        logits_list = self.gnn_forward(graph)
        logits = self.fc_forward(logits_list)
        return logits

    def training_step(self, batch, batch_idx):
        """Lightning calls this inside the training loop."""
        # Make a forward pass through the network for a batch of input graphs
        graph, labels = batch[0], batch[1].squeeze()

        # Forward propagate with network layers
        logits = self.shared_step(graph)

        # Calculate training loss
        loss = self.loss_fn(logits, labels)  # Calculate loss

        # Log training step metric(s)
        self.log(f'train_mse', loss, sync_dist=True)

        return {
            'loss': loss
        }

    def validation_step(self, batch, batch_idx):
        """Lightning calls this inside the validation loop."""
        # Make a forward pass through the network for a batch of input graphs
        graph, labels = batch[0], batch[1].squeeze()

        # Forward propagate with network layers
        logits = self.shared_step(graph)

        # Calculate validation loss
        loss = self.loss_fn(logits, labels)  # Calculate loss

        # Log validation step metric(s)
        self.log(f'val_mse', loss, sync_dist=True)

        return {
            'loss': loss
        }

    def test_step(self, batch, batch_idx):
        """Lightning calls this inside the testing loop."""
        # Make a forward pass through the network for a batch of input graphs
        graph, labels = batch[0], batch[1].squeeze()

        # Forward propagate with network layers
        logits = self.shared_step(graph)

        # Calculate test loss
        loss = self.loss_fn(logits, labels)  # Calculate loss

        # Log test step metric(s)
        self.log(f'test_mse', loss, sync_dist=True)

        return {
            'loss': loss
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Lightning calls this inside the predict loop."""
        graph = batch  # Make predictions for a batch of input graphs
        logits = self.shared_step(graph)  # Forward propagate with network layers
        return logits

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-8, verbose=True),
                "monitor": self.metric_to_track,
            }
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # -----------------
        # Model arguments
        # -----------------
        parser.add_argument('--num_gnn_layers', type=int, default=2, help='Number of GNN layers to apply')
        parser.add_argument('--num_gnn_hidden_channels', type=int, default=128,
                            help='Dimensionality of GNN filters (for nodes and edges alike after embedding)')
        parser.add_argument('--num_gnn_attention_heads', type=int, default=4,
                            help='How many multi-head GNN attention blocks to run in parallel')
        return parser

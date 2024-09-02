import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.encoder.custom_oebb_node_encoder import CustomOebbNodeEncoder
from torch_geometric.nn import GATConv




@register_network('custom_gnn_categorical')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        emb_dim_category = cfg.model.emb_dim_category
        emb_dim_operator_class = cfg.model.emb_dim_operator_class


        ################## NEW PART #############
        # Initialize the custom node encoder with the required dimensions and counts
        self.custom_encoder = CustomOebbNodeEncoder(
            emb_dim_category=emb_dim_category ,
            emb_dim_operator_class=emb_dim_operator_class ,
            num_categories= cfg.model.num_categories,
            num_operator_classes= cfg.model.num_operator_classes
        )
        
        dim_in = dim_in - 2 + emb_dim_category + emb_dim_operator_class
        ################## NEW PART #############

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in



        """
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        """

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual))
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'gatconv':
            return GATConv
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):

        """

        print("Batch before custom encoder:")
        print("batch.x shape:", batch.x.shape)
        print("batch.x:", batch.x)
        print("batch.edge_index shape:", batch.edge_index.shape)
        unique_operator_classes = torch.unique(batch.x[:, 1])
        print("Unique operator classes:", unique_operator_classes)

        """
        

        # Apply the custom encoder to 'category' and 'operator_class'
        batch = self.custom_encoder(batch)

        """

        print("Batch after custom encoder:")
        print("batch.x shape:", batch.x.shape)
        print("batch.x:", batch.x)
        print("batch.edge_index shape:", batch.edge_index.shape)

        
        """
    

        # Apply the FeatureEncoder to the updated batch
        batch = self.encoder(batch)

        if cfg.gnn.layers_pre_mp > 0:
            batch = self.pre_mp(batch)

        for layer in self.gnn_layers:
            batch = layer(batch)

        batch = self.post_mp(batch)
        
        return batch

from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    cfg.train.pkl_filepath = "___"


    cfg.model.emb_dim_category =  5
    cfg.model.emb_dim_operator_class = 5
    cfg.model.num_operator_classes = 60
    cfg.model.num_categories = 30
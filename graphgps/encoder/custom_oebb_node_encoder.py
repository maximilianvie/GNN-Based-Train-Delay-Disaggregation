import torch
from torch_geometric.graphgym.register import register_node_encoder

@register_node_encoder('custom_oebb_node_encoder')
class CustomOebbNodeEncoder(torch.nn.Module):
    """
    Custom encoder for node features, embedding categorical features 'category' and 'operator_class'.
    Parameters:
    emb_dim_category - Embedding dimension for 'category'
    emb_dim_operator_class - Embedding dimension for 'operator_class'
    num_categories - Number of unique categories
    num_operator_classes - Number of unique operator classes
    """

    def __init__(self, emb_dim_category, emb_dim_operator_class, num_categories, num_operator_classes):
        super().__init__()

        
        
        # Embedding for 'category'
        self.category_embedding = torch.nn.Embedding(num_categories, emb_dim_category)
        torch.nn.init.xavier_uniform_(self.category_embedding.weight.data)
        
        # Embedding for 'operator_class'
        self.operator_class_embedding = torch.nn.Embedding(num_operator_classes, emb_dim_operator_class)
        torch.nn.init.xavier_uniform_(self.operator_class_embedding.weight.data)

    
    def forward(self, batch):
        # Assuming 'category' and 'operator_class' are the first two features
        category = batch.x[:, 0].long()  # Extract 'category' feature
        operator_class = batch.x[:, 1].long()  # Extract 'operator_class' feature
        
        # Embed 'category' and 'operator_class'
        category_embedded = self.category_embedding(category)
        operator_class_embedded = self.operator_class_embedding(operator_class)

        
        # Concatenate the embeddings with the rest of the node features
        rest_features = batch.x[:, 2:]  # Adjust the slicing based on your feature ordering
        batch.x = torch.cat([category_embedded, operator_class_embedded, rest_features], dim=1)
        
        return batch

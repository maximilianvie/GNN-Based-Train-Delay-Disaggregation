import os
import pickle
import torch
from torch_geometric.data import InMemoryDataset
import random
from collections import Counter
from torch_geometric.graphgym.config import cfg

class custom_oebb(InMemoryDataset):
    def __init__(self, root, pickle_file, mode, transform=None, pre_transform=None, pre_filter=None):

        print("Entering __init__ method")
        self.mode = mode

        print(mode)
        self.pickle_file = pickle_file
        print(pickle_file)

        super().__init__(root, transform, pre_transform, pre_filter)

        print("Printing the processed paths")
        print(self.processed_paths[0])

        self.data, self.slices = torch.load(self.processed_paths[0])

   
    @property
    def raw_file_names(self):
        return [self.pickle_file]

    @property
    def processed_file_names(self):
        return ['data.pt', 'norm_params.pt', 'pre_filter.pt', 'pre_transform.pt']
    
    def reassign_file_types(self, data_list):

        # Set the seed for reproducibility
        random.seed(1234)

        num_objects = len(data_list)
        indices = list(range(num_objects))
        random.shuffle(indices)
        
        num_train = int(0.8 * num_objects)
        num_test = int(0.1 * num_objects)
        

        for i in indices[:num_train]:
            data_list[i]._store['set_type'] = 0  # Train
        for i in indices[num_train:num_train + num_test]:
            data_list[i]._store['set_type'] = 1  # Test
        for i in indices[num_train + num_test:]:
            data_list[i]._store['set_type'] = 2  # Validation
        
        return data_list

    def process(self):
        print("Entering process method")
        
        with open(os.path.join(self.raw_dir, self.pickle_file), 'rb') as f:
            data_list = pickle.load(f)


        # Reassign file types randomly with an 80-10-10 split
        if self.mode != "inference-only":

            print("#################################################")
            print("Reassign file types randomly with an 80-10-10 split")
            # Reassign file types randomly with an 80-10-10 split
            data_list = self.reassign_file_types(data_list)
            print("Finished reassigning file types")
            print("#################################################")

            # Collect the set types
            set_types = [data._store['set_type'] for data in data_list]

            # Count the occurrences of each set type
            counts = Counter(set_types)

            # Print the distribution of set types
            print("Distribution of set types:")
            print(f"Train (0): {counts[0]}")
            print(f"Test (1): {counts[1]}")
            print(f"Validation (2): {counts[2]}")



        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]



        global_node_id = 0
        node_id_list = []
        node_features_list = []
        node_labels_list = []

        # Initialize lists to store training features and labels
        train_features = []
        train_labels = []

        # Apply transformations and create masks
        for data in data_list:

            # Store node identifiers separately
            num_nodes = data.num_nodes

            # We are only interested in the first label

            if(cfg.dataset.task_type == "classification"):
                data.y = data.y[:, 0].long()
            else:
                data.y = data.y[:, 1].long()


            # Setting the Train - Test - Val Masks for inference 
            if self.mode == "inference-only":

                node_ids = torch.arange(global_node_id, global_node_id + num_nodes, dtype=torch.long)
                node_id_list.append(node_ids)
                global_node_id += num_nodes

                node_features_list.append(data.x)
                node_labels_list.append(data.y)

                # only the tesk mask gets activated because we want to run inference
                data.test_mask = torch.ones(num_nodes, dtype=torch.bool)
                data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)

            # Setting the Train - Test - Val Masks for training
            else:

                # Create masks based on set_type
                data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

                for i in range(num_nodes):
                    if data.set_type == 0:
                        data.train_mask[i] = True
                    elif data.set_type == 2:
                        data.val_mask[i] = True
                    elif data.set_type == 1:
                        data.test_mask[i] = True


                """ For debugging the masks: 
                print("#################################################")
                print("Printing information about mask")

                print(data.train_mask.shape)
               
                print(f"Set Type: {data.set_type}")
                print(f"Number of Nodes: {data.num_nodes}")

                """

        ###################################################################################
        ####### Getting the feature and label means and std from TRAIN DATA
        ####### They are used for scaling the data for training AND inference
        ###################################################################################

        if self.mode != "inference-only":

            # This part is only for getting the normalization parameters
            # we only want to compute it based on the train data and only the non supernodes 
            for data in data_list:

                # Filter rows where the last feature is not 1 (is_supernode equals 0)
                non_supernode_mask = (data.x[:, -1] == 0)
                non_supernode_train_mask = non_supernode_mask & data.train_mask  # Apply the mask on top of the train_mask

                # Select features from longitude to number_of_traction_units
                start_idx = 2  # Index of longitude feature
                end_idx = 10  # Index of number_of_traction_units feature (upper limit is exclusive)

                train_features.append(data.x[non_supernode_train_mask, start_idx:end_idx])

                # Filter labels based on nodes_to_classify condition
                nodes_to_classify_mask = (data.x[:, -4] == 1)
                valid_labels_mask = non_supernode_train_mask & nodes_to_classify_mask

                # Print the number of files for each mask
                print("Number of files for non-supernode mask:", non_supernode_train_mask.sum().item())
                print("Number of files for combined mask:", valid_labels_mask.sum().item())

                train_labels.append(data.y[valid_labels_mask])
 

            # Concatenate all training features and labels
            train_features = torch.cat(train_features, dim=0).float()  # Convert to float
            train_labels = torch.cat(train_labels, dim=0).float()  # Convert to float


            # Calculate normalization parameters for features
            feature_means = train_features.mean(dim=0)
            feature_stds = train_features.std(dim=0)


            print("#################################################")
            print("Printing the feature means")
            print(feature_means)

            # Calculate normalization parameters for labels
            label_means = train_labels.mean(dim=0)
            label_stds = train_labels.std(dim=0)

            print("label mean")
            print(label_means)
            print("label std")
            print(label_stds)



        ################ Exporting Inference Data (features and labels if available) ################
        ################ And loading the parameters for scaling from previous training ##############
        else: 

            filtered_node_features_list = []
            filtered_node_labels_list = []

            for features, labels in zip(node_features_list, node_labels_list):

                # Append the filtered features and labels to the respective lists
                filtered_node_features_list.append(features)
                filtered_node_labels_list.append(labels)

            
            # Save the filtered features and labels
            torch.save(filtered_node_features_list, os.path.join(self.processed_dir, 'node_features.pt'))
            torch.save(filtered_node_labels_list, os.path.join(self.processed_dir, 'node_labels.pt'))

            # Load normalization parameters from 'norm_params.pt'
            working_dir = os.getcwd()
            norm_params_path = os.path.join(working_dir, 'norm_params.pt')

            norm_params = torch.load(norm_params_path)
            feature_means = norm_params['feature_means']
            feature_stds = norm_params['feature_stds']
            label_means = norm_params['label_means']
            label_stds = norm_params['label_stds']

        total_supernodes = 0


        ####################################################################################
        ################ Normalize all data (either training or inference)
        ####################################################################################
        for data in data_list:
            # Normalize features from longitude to number_of_traction_units
            start_idx = 2  # Index of longitude feature
            end_idx = 10  # Index of number_of_traction_units feature (upper limit is exclusive)

            data.x[:, start_idx:end_idx] = (data.x[:, start_idx:end_idx] - feature_means) / feature_stds

            
            if(cfg.dataset.task_type != "classification"):
                data.y = (data.y - label_means) / label_stds
                      

            # Drop the features train_number and file_number
            # Feature order before dropping


            """ 
            0 category, 1 operator_class, 2 longitude,3 latitude,4 trainpart_weight,
            5 departure_delay_seconds, 6 arrival_delay_seconds, 7 delay_change, 8 train_unique_sequence,
            9 number_of_traction_units, 10 sin_departure, 11 cos_departure,
            12 sin_arrival, 13 cos_arrival, 14 day_value, 15 encoded_freight, 
            16 encoded_ocp_type, 17 nodes_to_train_on, 
            18 train_number, 19  file_number, 20 is_supernode
            """

            selected_indices = [5, 6, 8, 15, 16, 17, 20]

        
            # Select the specified columns from data.x
            data.x = data.x[:, selected_indices]

            # Count the occurrences of value 1 in the last column (is_supernode)
            total_supernodes += (data.x[:, -1] == 1).sum().item()

            #data.x = torch.cat((data.x[:, :17], data.x[:, -1:]), dim=1)

        print("#################################################")
        print("Printing first row of the normalized features")
        print(data_list[0].x[:10])

        print(f"Total number of supernodes: {total_supernodes}")


        # Collate the dataset
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])


        if self.mode != "inference-only":

            # Save normalization parameters
            norm_params = {
                'feature_means': feature_means,
                'feature_stds': feature_stds,
                'label_means': label_means,
                'label_stds': label_stds
            }
            torch.save(norm_params, os.path.join(self.processed_dir, 'norm_params.pt'))


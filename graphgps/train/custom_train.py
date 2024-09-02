import logging
import time
import csv
import os

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):

    model.train()
    optimizer.zero_grad()
    time_start = time.time()

    for iter, batch in enumerate(loader):

        batch.split = 'train'
        batch.to(torch.device(cfg.accelerator))

        
        if hasattr(batch, 'train_mask'):
            train_mask = batch.train_mask
        else:

            # If there's no specific training mask, raise an error
            raise ValueError("No specific training mask provided; this is required for training.")

            # If there's no specific training mask, consider all nodes for training
            #train_mask = torch.ones(batch.num_nodes, dtype=torch.bool)

        # Now apply the supernode and >60 seconds mask, but only for nodes in the training mask
        nodes_to_predict = batch.x[train_mask, -2] == 1

        # Check if there are any training nodes left after applying the supernode mask
        if not nodes_to_predict.any():
            # print(f"Skipping batch {iter} due to no training data. (likely test or validation batch)")
            continue  # Skip this batch

        pred, true = model(batch)


        # Apply the mask to predictions and ground truth
        pred = pred[nodes_to_predict]
        true = true[nodes_to_predict]

 
        
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)

        
        if torch.isnan(loss).any():
            print(f"Warning: NaN loss encountered on batch {iter}.")
            continue  # Optionally skip this batch or handle NaN loss differently


        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()

    


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()

    all_y_true = []
    all_y_pred = []
    all_features = []

    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))

        features = batch.x

        if hasattr(batch, f'{split}_mask'):
            eval_mask = getattr(batch, f'{split}_mask')
        else:
            #eval_mask = torch.ones(batch.num_nodes, dtype=torch.bool)

            raise ValueError("WARNING!!!! - No Mask Used for Evaluation!")

        if eval_mask.sum() > 0:

            nodes_to_predict = (batch.x[eval_mask, -2] == 1)

            if cfg.gnn.head == 'inductive_edge':
                pred, true, extra_stats = model(batch)
            else:
                pred, true = model(batch)
                extra_stats = {}


            # For inference we want to predict on all nodes but in the end only use certain probabilities 
            if(cfg.train.mode != "inference-only"):
                
                pred = pred[nodes_to_predict]
                true = true[nodes_to_predict]
                features = features[nodes_to_predict] 


            if cfg.dataset.name == 'ogbg-code2':
                loss, pred_score = subtoken_cross_entropy(pred, true)
                _true = true
                _pred = pred_score
            else:
                loss, pred_score = compute_loss(pred, true)
                _true = true.detach().to('cpu', non_blocking=True)
                _pred = pred_score.detach().to('cpu', non_blocking=True)

            logger.update_stats(true=_true,
                                pred=_pred,
                                loss=loss.detach().cpu().item(),
                                lr=0, time_used=time.time() - time_start,
                                params=cfg.params,
                                dataset_name=cfg.dataset.name,
                                **extra_stats)
            time_start = time.time()

            all_y_true.append(_true.numpy())
            all_y_pred.append(_pred.numpy())
            all_features.append(features.cpu().numpy()) 

    if len(all_y_true) > 0 and len(all_y_pred) > 0:
        aggregated_y_true = np.concatenate(all_y_true, axis=0)
        aggregated_y_pred = np.concatenate(all_y_pred, axis=0)
        aggregated_features = np.concatenate(all_features, axis=0)
        return aggregated_y_true, aggregated_y_pred, aggregated_features
    else:
        return None, None, None
    



@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0

    agg_true = np.empty(0)
    agg_pred = np.empty(0)


    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))


                if(cur_epoch == cfg.optim.max_epoch - 1):
                    agg_true, agg_pred, _ = eval_epoch(loggers[i], loaders[i], model,split=split_names[i - 1])

                    agg_true_plots, agg_pred_plots, _ = eval_epoch(loggers[i], loaders[i], model,split='test')

                    


        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, None, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, None, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
                        
            
                            
      
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")

    # Only compute and log the ROC curve and confusion matrix for the test dataset after the final epoch
    if cfg.dataset.task_type == 'classification':

        # Create a DataFrame to store the values
        data = {
            'Actual Values': agg_true_plots,
            'Predicted Values': agg_pred_plots
        }

        df = pd.DataFrame(data)

        # Save DataFrame to CSV
        csv_path = 'agg_values_classification.csv'
        df.to_csv(csv_path, index=False)



        fpr, tpr, _ = roc_curve(agg_true_plots, agg_pred_plots)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_plot_path = 'roc_curve.png'
        plt.savefig(roc_plot_path)
        plt.close()

        cm = confusion_matrix(agg_true_plots, (agg_pred_plots >= 0.5).astype(int))
        cm_plot_path = 'confusion_matrix.png'
        fig, ax = plt.subplots()
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([0, 1])
        ax.set_yticklabels([0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(cm_plot_path)
        plt.close()

        if cfg.wandb.use:
            run.log({"roc_curve": wandb.Image(roc_plot_path)})
            run.log({"confusion_matrix": wandb.Image(cm_plot_path)})


    # Call this function after the final epoch
    if cfg.dataset.task_type == 'regression':
        agg_true_values = agg_true_plots
        agg_pred_values = agg_pred_plots
        # Residual Plot
        residuals = np.array(agg_true_values) - np.array(agg_pred_values)


        # Create a DataFrame to store the values
        data = {
            'Actual Values': agg_true_values,
            'Predicted Values': agg_pred_values,
            'Residuals': residuals
        }

        df = pd.DataFrame(data)

        # Save DataFrame to CSV
        csv_path = 'agg_values.csv'
        df.to_csv(csv_path, index=False)
        
        plt.figure()
        plt.scatter(agg_pred_values, residuals, color='blue', s=10, label='Residuals')
        plt.axhline(y=0, color='red', linestyle='--', lw=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.legend(loc="upper right")
        residual_plot_path = 'residual_plot.png'
        plt.savefig(residual_plot_path)
        plt.close()

        # Predicted vs Actual Plot
        plt.figure()
        plt.scatter(agg_true_values, agg_pred_values, color='green', s=10, label='Predicted vs Actual')
        plt.plot([min(agg_true_values), max(agg_true_values)], [min(agg_true_values), max(agg_true_values)], color='red', linestyle='--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Plot')
        plt.legend(loc="upper left")
        pred_vs_actual_plot_path = 'pred_vs_actual_plot.png'
        plt.savefig(pred_vs_actual_plot_path)
        plt.close()








    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


def map_results_to_nodes(all_y_true, all_y_pred, features_in):

    # Load normalization parameters
    norm_params = torch.load(os.path.join('norm_params.pt'))
    label_means = norm_params['label_means'].numpy()  # Convert to numpy
    label_stds = norm_params['label_stds'].numpy()    # Convert to numpy
    feature_means = norm_params['feature_means'].numpy()  # Convert to numpy
    feature_stds = norm_params['feature_stds'].numpy()    # Convert to numpy

    selected_indices = [3]
    features_in = features_in[:, 0]

    print(feature_means)

    # Extract the selected indices from the normalization parameters
    selected_feature_means = feature_means[selected_indices]

    print("selected_feature_means")
    print(selected_feature_means)
    selected_feature_stds = feature_stds[selected_indices]

    # Denormalize only the selected features
    selected_features = features_in
    denormalized_features = selected_features * selected_feature_stds + selected_feature_means

    # Only keep the first column of the denormalized selected features
    departure_seconds_from_feature_in = denormalized_features


    """ 0 category, 1 operator_class, 2 longitude,3 latitude, 4 trainpart_weight,
    5 departure_delay_seconds, 6 arrival_delay_seconds, 7 train_unique_sequence,
    8 number_of_traction_units, 9 sin_departure, 10 cos_departure,
    11 sin_arrival, 12 cos_arrival, 13 day_value,  14 encoded_freight, 
    15 encoded_ocp_type, 16 nodes_to_train_on, 
    17 train_number, 18 file_number,19 is_supernode
    """


    node_features = torch.load(os.path.join('datasets/processed/node_features.pt'))
    node_labels = torch.load(os.path.join('datasets/processed/node_labels.pt'))

    print("length node_features######################")
    print(len(node_features))


    train_y_true = np.concatenate(all_y_true).ravel() if isinstance(all_y_true[0], np.ndarray) else all_y_true
    train_y_pred = np.concatenate(all_y_pred).ravel() if isinstance(all_y_pred[0], np.ndarray) else all_y_pred

    normalized_y_true = train_y_true
    normalized_y_pred = train_y_pred

    if(cfg.dataset.task_type == "regression"):
        # Denormalize y_true and y_pred
        train_y_true = (train_y_true * label_stds) + label_means
        train_y_pred = (train_y_pred * label_stds) + label_means

    # Print the lengths of the normalized y true and y pred
    print("length normalized_y_true######################")
    print(len(normalized_y_true))

    print("length normalized_y_pred######################")
    print(len(normalized_y_pred))
    mapped_results = []
    prediction_index = 0
    debug_counter = 0  # Counter to limit the number of print statements


    for graph_features, graph_labels in zip(node_features, node_labels):
        

        for feature, label in zip(graph_features, graph_labels):

            
            norm_true_val = normalized_y_true[prediction_index]
            norm_pred_val = normalized_y_pred[prediction_index]

            true_val = train_y_true[prediction_index]
            pred_val = train_y_pred[prediction_index]

            departure_seconds_in_model  = departure_seconds_from_feature_in[prediction_index]

            # Adjust handling for NumPy arrays
            true_val_list = true_val.tolist() if isinstance(true_val, np.ndarray) and true_val.size > 1 else true_val
            pred_val_list = pred_val.tolist() if isinstance(pred_val, np.ndarray) and pred_val.size > 1 else pred_val
           
 

            feature_list = feature.tolist() if feature.nelement() > 1 else feature.item()
            label_list = label.tolist() if label.nelement() > 1 else label.item()


            """ 
            0 category, 1 operator_class, 2 longitude,3 latitude,4 trainpart_weight,
            5 departure_delay_seconds, 6 arrival_delay_seconds, 7 delay_change, 8 train_unique_sequence,
            9 number_of_traction_units, 10 sin_departure, 11 cos_departure,
            12 sin_arrival, 13 cos_arrival, 14 day_value, 15 encoded_freight, 
            16 encoded_ocp_type, 17 nodes_to_train_on, 
            18 train_number, 19  file_number, 20 is_supernode
            """

            # Print statements (limited to the first 5 rows)
            if debug_counter < 5:
                print("Mapped Feature:", feature_list)
                print("Mapped Label:", label_list)
                print("Mapped True Value:", true_val_list)
                print("Mapped Predicted Value:", pred_val_list)
                print("Feature list:")
                print("Departure Delay Seconds:", feature_list[5])
                print("Arrival Delay Seconds:", feature_list[6])
                print("Train Unique Sequence:", feature_list[8])
                print("Encoded Freight:", feature_list[15])
                print("Encoded OCP Type:", feature_list[16])
                print("Nodes to train on:", feature_list[17])

                print("Train Number:", feature_list[18])
                print("File Number:", feature_list[19])
                print("Is Supernode:", feature_list[20])
                debug_counter += 1



  


            departure_delay_seconds_total = feature_list[5]
            arrival_delay_seconds_total = feature_list[6]

            
            TrainNumber = feature_list[18]
            TrainSequence = feature_list[8]
            FileNumber = feature_list[19]
            NodesToTrainOn = feature_list[17]
            IsSupernode = feature_list[20]



            rounded_values = [round(x, 2) if isinstance(x, float) else x for x in [FileNumber, TrainNumber, TrainSequence, label_list, true_val_list, pred_val_list, norm_true_val, norm_pred_val, arrival_delay_seconds_total, departure_delay_seconds_total, departure_seconds_in_model, NodesToTrainOn, IsSupernode]]
            mapped_results.append(tuple(rounded_values))
            
            prediction_index += 1
            

    return mapped_results


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):


    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """

    all_y_true = []
    all_y_pred = []



    num_splits = 1
    
    perf = [[] for _ in range(num_splits)]
    start_time = time.perf_counter()



    # 2 for test loader
    y_true, y_pred, features_in = eval_epoch(loggers[2], loaders[2], model, split="test")

    print("y_pred- length")

    print(len(y_pred))

    perf[0].append(loggers[2].write_epoch(0))
    all_y_true.append(y_true)  
    all_y_pred.append(y_pred)


    mapped_results = map_results_to_nodes(all_y_true, all_y_pred, features_in)

 
    with open('mapped_inference_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['FileNumber', 'TrainNumber', 'TrainSequence', 'label_verify', 'y_true', 'y_pred', 'norm_y_true', 'norm_y_pred', 'arrival_delay_seconds_total', 'departure_delay_seconds_total', 'departure_seconds_in_model', 'nodes_to_train_on', 'is_Supernode'])
        for  FileNumber, TrainNumber, TrainSequence, label_list, true_val_list, pred_val_list, normalized_y_true, normalized_y_pred, arrival_delay_seconds_total, departure_delay_seconds_total, departure_seconds_in_model, NodesToTrainOn, isSupernode in mapped_results:
            writer.writerow([FileNumber, TrainNumber, TrainSequence, label_list, true_val_list, pred_val_list, normalized_y_true, normalized_y_pred, arrival_delay_seconds_total, departure_delay_seconds_total, departure_seconds_in_model, NodesToTrainOn, isSupernode])

    best_epoch = 0
    best_train = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()

    

@ register_train('log-attn-weights')
def log_attn_weights(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to inference on the test set and log the attention
    weights in Transformer modules.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model (torch.nn.Module): GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    import os.path as osp
    from torch_geometric.loader.dataloader import DataLoader
    from graphgps.utils import unbatch, unbatch_edge_index

    start_time = time.perf_counter()

    # The last loader is a test set.
    l = loaders[-1]
    # To get a random sample, create a new loader that shuffles the test set.
    loader = DataLoader(l.dataset, batch_size=l.batch_size,
                        shuffle=True, num_workers=0)

    output = []
    # batch = next(iter(loader))  # Run one random batch.
    for b_index, batch in enumerate(loader):
        bsize = batch.batch.max().item() + 1  # Batch size.
        if len(output) >= 128:
            break
        print(f">> Batch {b_index}:")

        X_orig = unbatch(batch.x.cpu(), batch.batch.cpu())
        batch.to(torch.device(cfg.accelerator))
        model.eval()
        model(batch)

        # Unbatch to individual graphs.
        X = unbatch(batch.x.cpu(), batch.batch.cpu())
        edge_indices = unbatch_edge_index(batch.edge_index.cpu(),
                                          batch.batch.cpu())
        graphs = []
        for i in range(bsize):
            graphs.append({'num_nodes': len(X[i]),
                           'x_orig': X_orig[i],
                           'x_final': X[i],
                           'edge_index': edge_indices[i],
                           'attn_weights': []  # List with attn weights in layers from 0 to L-1.
                           })

        # Iterate through GPS layers and pull out stored attn weights.
        for l_i, (name, module) in enumerate(model.model.layers.named_children()):
            if hasattr(module, 'attn_weights'):
                print(l_i, name, module.attn_weights.shape)
                for g_i in range(bsize):
                    # Clip to the number of nodes in this graph.
                    # num_nodes = graphs[g_i]['num_nodes']
                    # aw = module.attn_weights[g_i, :num_nodes, :num_nodes]
                    aw = module.attn_weights[g_i]
                    graphs[g_i]['attn_weights'].append(aw.cpu())
        output += graphs

    logging.info(
        f"[*] Collected a total of {len(output)} graphs and their "
        f"attention weights for {len(output[0]['attn_weights'])} layers.")

    # Save the graphs and their attention stats.
    save_file = osp.join(cfg.run_dir, 'graph_attn_stats.pt')
    logging.info(f"Saving to file: {save_file}")
    torch.save(output, save_file)

    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def get_user_positive_items(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items

def RecallPrecision_ATk(groundTruth, r, k):
    """Computes recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()

def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

def get_metrics(rating, edge_index, exclude_edge_indices, k):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    
    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set ratings of excluded edges to large negative value
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return round(recall, 5), round(precision, 5), round(ndcg, 5)

def get_metrics_list(model, edge_index, exclude_edge_indices, k):
    
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    # get ratings between every user and item - shape is num users x num movies
    rating = torch.matmul(user_embedding, item_embedding.T)
    
    recalls = []
    precisions = []
    ndcgs = []
    
    for num in k:
        recall_num, precision_num, ndcg_num = get_metrics(rating, edge_index, exclude_edge_indices, num)
        recalls.append(recall_num)
        precisions.append(precision_num)
        ndcgs.append(ndcg_num)
        
    return recalls, precisions, ndcgs

def print_results(recalls, precisions, ndcgs, K):
    
    table_K = K.copy()
    table_K.insert(0, 'K')
    
    table_recalls = recalls.copy()
    table_recalls.insert(0, 'Recall@K')
    
    table_precisions = precisions.copy()
    table_precisions.insert(0, 'Precision@K')
    
    table_ndcgs = ndcgs.copy()
    table_ndcgs.insert(0, 'NDCG@K')
    
    table = [table_K, table_recalls, table_precisions, table_ndcgs]
    for row in table:
        print('| {:11} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} |'.format(*row))
        
def plot_train_val(train_losses, val_losses, iters_per_eval, name, n_iter, output_dir='results'):
    iters = [iter * iters_per_eval for iter in range(len(train_losses))]
    plt.plot(iters, train_losses, label='train')
    plt.plot(iters, val_losses, label='validation')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('training and validation loss curves')
    plt.legend()
    
    if not os.path.exists(f'{output_dir}/{name}_iter{n_iter}'):
        os.mkdir(f'{output_dir}/{name}_iter{n_iter}')
    
    plt.savefig(f'{output_dir}/{name}_iter{n_iter}/train_val_loss.png')
    
def save_results(recalls, precisions, ndcgs, K, name, n_iter, output_dir='results'):
    
    # save results CSV
    
    res_dict = {
        'K': K,
        'Recall@K': recalls,
        'Precision@K': precisions,
        'NDCG@K': ndcgs,
    }

    df = pd.DataFrame(res_dict)
    
    if not os.path.exists(f'{output_dir}/{name}_iter{n_iter}'):
        os.mkdir(f'{output_dir}/{name}_iter{n_iter}')
        
    df.to_csv(f'{output_dir}/{name}_iter{n_iter}/results.csv', index=False)
    
    # plot and save bar
    plot_dict={}
    for i, k in enumerate(K):
        plot_dict[k] = [recalls[i], precisions[i], ndcgs[i]]

    metrics = ['Recall@K', 'Precision@K', 'NDCG@K']

    _, ax = plt.subplots(layout='constrained')
    ax.set_ylim((0,1))

    x = np.arange(len(metrics))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    for metric, val in plot_dict.items():
        metric = str(metric)
        offset = width * multiplier
        rects = ax.bar(x + offset, val, width, label=metric)
        ax.bar_label(rects, padding=3, fmt='%.2f')
        multiplier += 1

    ax.set_xlabel('Metric Name')
    ax.set_ylabel('Value')
    ax.set_title(f'Results - Model: {name}, Iterations: {n_iter}')
    ax.set_xticks(x + width*2, metrics)
    ax.legend(title='K', ncols=5)
        
    plt.savefig(f'{output_dir}/{name}_iter{n_iter}/results.png')
    
def plot_val_metrics(val_recalls, val_precisions, val_ndcgs, K, iters_per_eval, name, n_iter, output_dir='results'):
    
    fig, axes = plt.subplots(1,3, figsize=(20,4))
    
    for ax in axes:
        ax.set_ylim((0,1))

    str_K = [str(x) for x in K]
    iters = [iter * iters_per_eval for iter in range(len(val_recalls))]
    
    axes[0].plot(iters, val_recalls, label=str_K)
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('value')
    axes[0].set_title('Recall@K')
    axes[0].legend()

    axes[1].plot(iters, val_precisions, label=str_K)
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('value')
    axes[1].set_title('Precision@K')
    axes[1].legend()

    axes[2].plot(iters, val_ndcgs, label=str_K)
    axes[2].set_xlabel('iteration')
    axes[2].set_ylabel('value')
    axes[2].set_title('NDCG@K')
    axes[2].legend()

    fig.suptitle('Metrics across iterations')
    
    if not os.path.exists(f'{output_dir}/{name}_iter{n_iter}'):
        os.mkdir(f'{output_dir}/{name}_iter{n_iter}')
    
    plt.savefig(f'{output_dir}/{name}_iter{n_iter}/val_metrics.png')
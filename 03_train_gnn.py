import os
import sys
import argparse
import pathlib
import numpy as np


def pretrain(policy, pretrain_loader):
    policy.pre_train_init()
    i = 0
    while True:
        for batch in pretrain_loader:
            batch.to(device)
            if not policy.pre_train(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features):
                break

        if policy.pre_train_next() is None:
            break
        i += 1
    return i


def process(policy, data_loader, top_k=[1, 3, 5, 10], optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))
    mean_entropy = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            logits = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            cross_entropy_loss = F.cross_entropy(logits, batch.candidate_choices, reduction='mean')
            entropy = (-F.softmax(logits, dim=-1)*F.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss = cross_entropy_loss - entropy_bonus*entropy

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            kacc = []
            for k in top_k:
                if logits.size()[-1] < k:
                    kacc.append(1.0)
                    continue
                pred_top_k = logits.topk(k).indices
                pred_top_k_true_scores = true_scores.gather(-1, pred_top_k)
                accuracy = (pred_top_k_true_scores == true_bestscore).any(dim=-1).float().mean().item()
                kacc.append(accuracy)
            kacc = np.asarray(kacc)
            mean_loss += cross_entropy_loss.item() * batch.num_graphs
            mean_entropy += entropy.item() * batch.num_graphs
            mean_kacc += kacc * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed
    mean_entropy /= n_samples_processed
    return mean_loss, mean_kacc, mean_entropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'mknapsack'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    max_epochs = 1000
    batch_size = 32
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr = 1e-3
    entropy_bonus = 0.0
    top_k = [1, 3, 5, 10]

    problem_folders = {
        'setcover': 'setcover/500r_1000c_0.05d',
        'cauctions': 'cauctions/100_500',
        'facilities': 'facilities/100_100_5',
        'indset': 'indset/500_4',
        'mknapsack': 'mknapsack/100_6',
    }
    problem_folder = problem_folders[args.problem]
    running_dir = f"model/{args.problem}/{args.seed}"
    os.makedirs(running_dir, exist_ok=True)

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from utilities import log, pad_tensor, GraphDataset, Scheduler
    sys.path.insert(0, os.path.abspath(f'model'))
    from model import GNNPolicy

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    ### LOG ###
    logfile = os.path.join(running_dir, 'train_log.txt')
    if os.path.exists(logfile):
        os.remove(logfile)

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"entropy bonus: {entropy_bonus}", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)


    policy = GNNPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = Scheduler(optimizer, mode='min', patience=10, factor=0.2, verbose=True)

    train_files = [str(file) for file in (pathlib.Path(f'data/samples')/problem_folder/'train').glob('sample_*.pkl')]
    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    valid_files = [str(file) for file in (pathlib.Path(f'data/samples')/problem_folder/'valid').glob('sample_*.pkl')]

    pretrain_data = GraphDataset(pretrain_files)
    pretrain_loader = torch_geometric.loader.DataLoader(pretrain_data, pretrain_batch_size, shuffle=False)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, valid_batch_size, shuffle=False)

    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        if epoch == 0:
            n = pretrain(policy, pretrain_loader)
            log(f"PRETRAINED {n} LAYERS", logfile)
        else:
            epoch_train_files = rng.choice(train_files, int(np.floor(10000/batch_size))*batch_size, replace=True)
            train_data = GraphDataset(epoch_train_files)
            train_loader = torch_geometric.data.DataLoader(train_data, batch_size, shuffle=True)
            train_loss, train_kacc, entropy = process(policy, train_loader, top_k, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # TEST
        valid_loss, valid_kacc, entropy = process(policy, valid_loader, top_k, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        scheduler.step(valid_loss)
        if scheduler.num_bad_epochs == 0:
            torch.save(policy.state_dict(), pathlib.Path(running_dir)/'train_params.pkl')
            log(f"  best model so far", logfile)
        elif scheduler.num_bad_epochs == 10:
            log(f"  10 epochs without improvement, decreasing learning rate", logfile)
        elif scheduler.num_bad_epochs == 20:
            log(f"  20 epochs without improvement, early stopping", logfile)
            break

    policy.load_state_dict(torch.load(pathlib.Path(running_dir)/'train_params.pkl'))
    valid_loss, valid_kacc, entropy = process(policy, valid_loader, top_k, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

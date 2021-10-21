import os
import sys
import importlib
import argparse
import csv
import numpy as np
import time
import pickle

import ecole
import pyscipopt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    result_file = f"{args.problem}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    instances = []
    seeds = [0]
    internal_branchers = ['relpscost']
    gnn_models = ['supervised'] # Can be supervised
    time_limit = 3600

    if args.problem == 'setcover':
        instances += [{'type': 'small', 'path': f"data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]

    elif args.problem == 'cauctions':
        instances += [{'type': 'small', 'path': f"data/instances/cauctions/transfer_100_500/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/cauctions/transfer_200_1000/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/cauctions/transfer_300_1500/instance_{i+1}.lp"} for i in range(20)]

    elif args.problem == 'facilities':
        instances += [{'type': 'small', 'path': f"data/instances/facilities/transfer_100_100_5/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/facilities/transfer_200_100_5/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/facilities/transfer_400_100_5/instance_{i+1}.lp"} for i in range(20)]

    elif args.problem == 'indset':
        instances += [{'type': 'small', 'path': f"data/instances/indset/transfer_500_4/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/indset/transfer_1000_4/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/indset/transfer_1500_4/instance_{i+1}.lp"} for i in range(20)]

    else:
        raise NotImplementedError

    branching_policies = []

    # SCIP internal brancher baselines
    for brancher in internal_branchers:
        for seed in seeds:
            branching_policies.append({
                    'type': 'internal',
                    'name': brancher,
                    'seed': seed,
             })
    # GNN models
    for model in gnn_models:
        for seed in seeds:
            branching_policies.append({
                'type': 'gnn',
                'name': model,
                'seed': seed,
            })

    print(f"problem: {args.problem}")
    print(f"gpu: {args.gpu}")
    print(f"time limit: {time_limit} s")

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    import torch
    from model.model import GNNPolicy

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}
    loaded_calls = {}
    for policy in branching_policies:
        if policy['type'] == 'gnn':
            if policy['name'] not in loaded_models:
                ### MODEL LOADING ###
                model = GNNPolicy().to(device)
                if policy['name'] == 'supervised':
                    model.load_state_dict(torch.load(f"model/{args.problem}/{policy['seed']}/train_params.pkl"))
                else:
                    raise Exception(f"Unrecognized GNN policy {policy[name]}")
                loaded_models[policy['name']] = model

            policy['model'] = loaded_models[policy['name']]

    print("running SCIP...")

    fieldnames = [
        'policy',
        'seed',
        'type',
        'instance',
        'nnodes',
        'nlps',
        'stime',
        'gap',
        'status',
        'walltime',
        'proctime',
    ]
    os.makedirs('results', exist_ok=True)
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': time_limit,
                       'timing/clocktype': 1, 'branching/vanillafullstrong/idempotent': True}

    with open(f"results/{result_file}", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            for policy in branching_policies:
                if policy['type'] == 'internal':
                    # Run SCIP's default brancher
                    env = ecole.environment.Configuring(scip_params={**scip_parameters,
                                                        f"branching/{policy['name']}/priority": 9999999})
                    env.seed(policy['seed'])

                    walltime = time.perf_counter()
                    proctime = time.process_time()

                    env.reset(instance['path'])
                    _, _, _, _, _ = env.step({})

                    walltime = time.perf_counter() - walltime
                    proctime = time.process_time() - proctime

                elif policy['type'] == 'gnn':
                    # Run the GNN policy
                    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(),
                                                      scip_params=scip_parameters)
                    env.seed(policy['seed'])
                    torch.manual_seed(policy['seed'])

                    walltime = time.perf_counter()
                    proctime = time.process_time()

                    observation, action_set, _, done, _ = env.reset(instance['path'])
                    while not done:
                        with torch.no_grad():
                            observation = (torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                                           torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
                                           torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
                                           torch.from_numpy(observation.column_features.astype(np.float32)).to(device))

                            logits = policy['model'](*observation)
                            action = action_set[logits[action_set.astype(np.int64)].argmax()]
                            observation, action_set, _, done, _ = env.step(action)

                    walltime = time.perf_counter() - walltime
                    proctime = time.process_time() - proctime

                scip_model = env.model.as_pyscipopt()
                stime = scip_model.getSolvingTime()
                nnodes = scip_model.getNNodes()
                nlps = scip_model.getNLPs()
                gap = scip_model.getGap()
                status = scip_model.getStatus()

                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': policy['seed'],
                    'type': instance['type'],
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'stime': stime,
                    'gap': gap,
                    'status': status,
                    'walltime': walltime,
                    'proctime': proctime,
                })
                csvfile.flush()

                print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")

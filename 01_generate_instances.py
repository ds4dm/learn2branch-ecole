import argparse
import ecole
from pathlib import Path


def generate_setcovers(n, folder_name, nrows, ncols, dens, rng):
    lp_dir = f"setcover/{folder_name}_{nrows}r_{ncols}c_{dens}d"
    instances = ecole.instance.SetCoverGenerator(n_rows=nrows, n_cols=ncols, density=dens,rng=rng)
    generate_instances(instances, lp_dir, n)


def generate_indsets(n, folder_name, number_of_nodes, affinity, rng):
    lp_dir = f"indset/{folder_name}_{number_of_nodes}_{affinity}"
    barabasi_albert = ecole.instance.IndependentSetGenerator.GraphType.barabasi_albert
    instances = ecole.instance.IndependentSetGenerator(n_nodes=number_of_nodes, affinity=affinity, graph_type=barabasi_albert, rng=rng)
    generate_instances(instances, lp_dir, n)


def generate_cauctions(n, folder_name, number_of_items, number_of_bids, rng):
    lp_dir = f"cauctions/{folder_name}_{number_of_items}_{number_of_bids}"
    instances = ecole.instance.CombinatorialAuctionGenerator(n_items=number_of_items, n_bids=number_of_bids, add_item_prob=0.7, rng=rng)
    generate_instances(instances, lp_dir, n)


def generate_facilities(n, folder_name, number_of_customers, number_of_facilities, ratio, rng):
    lp_dir = f"facilities/{folder_name}_{number_of_customers}_{number_of_facilities}_{ratio}"
    instances = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=number_of_customers,  n_facilities=number_of_facilities,ratio=ratio, rng=rng)
    generate_instances(instances, lp_dir, n)


def generate_instances(instances, lp_dir, n):
    lp_dir = Path(f"data/instances")/lp_dir
    lp_dir.mkdir(parents=True)
    print(f"{n} instances in {lp_dir}")
    for i in range(n):
        instance = next(instances)
        filename = str(lp_dir/f"instance_{i+1}.lp")
        print(f'  generating file {filename} ...')
        instance.write_problem(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        default=0,
    )
    args = parser.parse_args()

    rng = ecole.spawn_random_generator()
    rng.seed(args.seed)

    
    if args.problem == 'setcover':
        nrows = 500
        ncols = 1000
        dens = 0.05
        max_coef = 100

        # train instances
        generate_setcovers(10000, "train", nrows, ncols, dens, rng)

        # validation instances
        generate_setcovers(2000, "valid", nrows, ncols, dens, rng)

        # small transfer instances
        generate_setcovers(100, "transfer", 500, ncols, dens, rng)

        # medium transfer instances
        generate_setcovers(100, "transfer", 1000, ncols, dens, rng)

        # big transfer instances
        generate_setcovers(100, "transfer", 2000, ncols, dens, rng)
        
        # test instances
        generate_setcovers(2000, "test", nrows, ncols, dens, rng)

        print('done.')
        

    elif args.problem == 'indset':
        number_of_nodes = 500
        affinity = 4

        # train instances
        generate_indsets(10000, "train", number_of_nodes, affinity, rng)

        # validation instances
        generate_indsets(2000, "valid", number_of_nodes, affinity, rng)

        # small transfer instances
        generate_indsets(100, "transfer", 500, affinity, rng)

        # medium transfer instances
        generate_indsets(100, "transfer", 1000, affinity, rng)

        # big transfer instances
        generate_indsets(100, "transfer", 1500, affinity, rng)

        # test instances
        generate_indsets(2000, "test", number_of_nodes, affinity, rng)

        print("done.")
        

    elif args.problem == 'cauctions':
        number_of_items = 100
        number_of_bids = 500

        # train instances
        generate_cauctions(10000, "train", number_of_items, number_of_bids, rng=rng)

        # validation instances
        generate_cauctions(2000, "valid", number_of_items, number_of_bids, rng=rng)

        # small transfer instances
        generate_cauctions(100, "transfer", 100, 500, rng=rng)

        # medium transfer instances
        generate_cauctions(100, "transfer", 200, 1000, rng=rng)

        # big transfer instances
        generate_cauctions(100, "transfer", 300, 1500, rng=rng)

        # test instances
        generate_cauctions(2000, "test", number_of_items, number_of_bids, rng=rng)

        print("done.")
        

    elif args.problem == 'facilities':
        number_of_customers = 100
        number_of_facilities = 100
        ratio = 5

        # train instances
        generate_facilities(10000, "train", number_of_customers, number_of_facilities, ratio, rng)

        # validation instances
        generate_facilities(2000, "valid", number_of_customers, number_of_facilities, ratio, rng)

        # small transfer instances
        generate_facilities(100, "transfer", 100, number_of_facilities, ratio, rng)
        
        # medium transfer instances
        generate_facilities(100, "transfer", 200, number_of_facilities, ratio, rng)

        # big transfer instances
        generate_facilities(100, "transfer", 400, number_of_facilities, ratio, rng)

        # test instances
        generate_facilities(2000, "test", number_of_customers, number_of_facilities, ratio, rng)


        print("done.")

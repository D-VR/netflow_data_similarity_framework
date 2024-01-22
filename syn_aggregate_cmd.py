import argparse
from src import aggregate_results
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load_path", required=True)
    parser.add_argument("-s", '--store_path', required=True)
    parser.add_argument("-i", '--ds_id', required=True)


    args = parser.parse_args()
    load_path = args.load_path
    store_path = args.store_path
    dataset_id = args.ds_id

    aggregate_results.aggregate_raw_results(load_path, store_path, dataset_id)
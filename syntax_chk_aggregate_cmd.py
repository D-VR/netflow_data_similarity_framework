import argparse
from src import syntax_check
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load_path", required=True)
    parser.add_argument("-s", '--store_path', required=True)
    #parser.add_argument("-i", '--ds_id', required=True)


    args = parser.parse_args()
    load_path = args.load_path
    store_path = args.store_path
    #dataset_id = args.ds_id

    syntax_check.aggreate_error_counts(load_path, store_path)

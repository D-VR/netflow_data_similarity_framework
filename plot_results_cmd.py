import argparse
from src import plot_results
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load_path", required=True)


    args = parser.parse_args()
    load_path = args.load_path
    
    plot_results.plot_results(load_path)
import argparse
from src import plot_results
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("-l", "--load_path", required=True)
    parser.add_argument("-p", "--prefix", required=True)
    parser.add_argument("-r", "--save_path_real", required=True)
    parser.add_argument("-s1d", "--save_path_syn1", required=True)
    parser.add_argument("-s1s", "--path_syntax_syn1", required=True)
    parser.add_argument("-s2d", "--save_path_syn2", required=True)
    parser.add_argument("-s2s", "--path_syntax_syn2", required=True)


    args = parser.parse_args()
    #load_path = args.load_path
    prefix = args.prefix
    save_path_real = args.save_path_real
    save_path_syn1 = args.save_path_syn1
    path_syntax_syn1 = args.path_syntax_syn1
    save_path_syn2 = args.save_path_syn2
    path_syntax_syn2 = args.path_syntax_syn2
    
    #plot_results.plot_results(load_path)
    plot_results.plot_results(prefix, save_path_real, save_path_syn1, path_syntax_syn1, save_path_syn2, path_syntax_syn2)

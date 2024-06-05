from dataload import *
from args import config
import warnings
from experiment import Experiment
from save import save_result
from dataload import datapro

warnings.filterwarnings("ignore")


def main(args, data):
    # fusion features
    entries = ['miRNA', 'disease']
    networks = dict()
    for entrie in entries:
        network = Experiment().MACFE(args, data, entrie)
        networks[entrie] = network
    m_d_matrix = datapro().read_csv(args.datapath + 'm_d.csv')
    m, n = m_d_matrix.shape
    original_adj = np.vstack((np.hstack((np.zeros(shape=(m, m), dtype=int), m_d_matrix)),
                              np.hstack((m_d_matrix.T, np.zeros(shape=(n, n), dtype=int)))))
    m_d_matrix = torch.FloatTensor(original_adj)
    networks['MDA'] = {'original_adj': original_adj, 'm_d_matrix': m_d_matrix}

    results, final_results = Experiment().train(args, networks)
    save_result(args, results, final_results)


if __name__ == "__main__":
    param = config()
    original_features = original_data(param)
    main(param, original_features)

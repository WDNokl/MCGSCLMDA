import argparse


def config():
    parser = argparse.ArgumentParser()
    # dataset
    # dataset
    # original data
    parser.add_argument('-miRNA', type=str, default=['m_ss', 'm_fs', 'm_gs'])
    parser.add_argument('-disease', type=str, default=['d_ss', 'd_ts', 'd_gs'])
    # MACFE
    parser.add_argument('-CFE_epoch', type=int, default=200)
    parser.add_argument('-CFE_layers', type=int, default=3, help='the numbers of autoencoder layers')
    parser.add_argument('-CFE_rate', type=float, default=0.08, choices=[0.1])
    parser.add_argument('-CFE_nets', type=int, default=3, help='the numbers of similarity network')
    parser.add_argument('-CFE_lr', type=float, default=[0.1, 0.05, 0.05])
    parser.add_argument('-CFE_gamma', type=float, default=0.9)
    parser.add_argument('-CFE_batch_size_miRNA', type=int, default=256)
    parser.add_argument('-CFE_miRNA_num', type=int, default=853)
    parser.add_argument('-CFE_hidden_miRNA', type=int, default=[700, 500, 853])
    parser.add_argument('-CFE_batch_size_disease', type=int, default=256)
    parser.add_argument('-CFE_disease_num', type=int, default=591)
    parser.add_argument('-CFE_hidden_disease', type=int, default=[500, 300, 591])

    parser.add_argument('-datapath', type=str, default='../Data/')
    parser.add_argument('-savepath', type=str, default='./result/')
    parser.add_argument('-m_num', type=int, default=853)
    parser.add_argument('-d_num', type=int, default=591)
    parser.add_argument('-m_emb', type=int, default=256)
    parser.add_argument('-d_emb', type=int, default=256)
    parser.add_argument('-gcn_layers', type=int, default=2)
    parser.add_argument('-view', type=int, default=3)
    # Experimental setting
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-k_fold', type=int, default=5, choices=[5, 10])
    parser.add_argument('-gpu', type=int, default=0)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=600)
    parser.add_argument('-lr', type=float, default=0.01, choices=[0.01])
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-emb_dim', type=int, default=256)
    parser.add_argument('-proj_dim', type=int, default=256)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2, choices=[0.2])
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.1, choices=[0.1])
    parser.add_argument('-dropedge_rate', type=float, default=0.2, choices=[0.2])

    # GSL Module
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('-gamma', type=float, default=0.9)

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=6)
    parser.add_argument('-lr_cls', type=float, default=0.001, choices=[0.001])
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=100)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=0.9999, choices=[0.9999])
    parser.add_argument('-c', type=int, default=1)

    # experiment
    parser.add_argument('-rate', type=int, default=0)
    parser.add_argument('-epoch', type=int, default=0)
    parser.add_argument('-ablation', type=int, default=0)
    # moco
    parser.add_argument('-m', type=float, default=0.999, choices=[0.999])
    parser.add_argument('-moco', type=int, default=1)
    parser.add_argument('-fusion', type=int, default=0, choices=[0, 1])  # 1代表consine相似性，0代表不融合
    parser.add_argument('-simview', type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()
    return args

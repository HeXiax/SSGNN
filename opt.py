import argparse

parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name',type=str,default='dblp')
parser.add_argument('--k',type=int, default=3)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--gamma1',type=float,default=1.0)
parser.add_argument('--gamma2',type=float,default=1.0)
parser.add_argument('--lambda1',type=float,default=20.0)
parser.add_argument('--mu',type=float,default=1000.0)
parser.add_argument('--iter',type=int,default=10)
parser.add_argument('--lambda2',type=float,default=0.1)
parser.add_argument('--lambda3',type=float,default=0.01)
parser.add_argument('--n_clusters',type=int,default=3)
parser.add_argument('--hidden1', type=int, default=500)
parser.add_argument('--hidden2', type=int, default=500)
parser.add_argument('--hidden3', type=int, default=2000)
parser.add_argument('--enc_1', type=int, default=128)
parser.add_argument('--enc_2', type=int, default=256)
parser.add_argument('--enc_3', type=int, default=512)
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--dec_1', type=int, default=512)
parser.add_argument('--dec_2', type=int, default=256)
parser.add_argument('--dec_3', type=int, default=128)
parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
parser.add_argument('--dropout',type=float,default=0.7)
parser.add_argument('--alpha',type=float,default=0.2)
args =parser.parse_args()
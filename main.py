from models.Informer.model import Informer, InformerStack
# from torchsummaryX import summary as summaryx
# from torchsummary import summary
# from models.Reformer.reformer_enc_dec import ReformerEncDec
import dataloader
import utils.utils as utils
from utils.metrics import MSE, MAE, RMSE
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from train import train, test, evaluate

plt.rcParams['agg.path.chunksize'] = 10000
parser = argparse.ArgumentParser(description='Time series forecasting')

parser.add_argument('--model', type=str, default='informer',
                    help='model of experiment, options: [informer, informerstack, informerlight(TBD), linformer]')

parser.add_argument('--dataset', type=str, help='dataset: e, stock', default=None)
parser.add_argument('--folder_name', type=str, help='exp-folder-name', default='')

parser.add_argument('--epochs', default=200, type=int, help='epoch (default: 200)')
parser.add_argument('--batch_size', default=1500, type=int, help='batch size (default: 1024)')
# parser.add_argument('--save-root', default='./exp-results-hidden32/', type=str, help='save root')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--hidden_size', default=32, type=int, help='hidden size (default: 128)')
parser.add_argument('--traindir', default='./data', type=str, help='train data path')
# parser.add_argument('--testdir',default = '/daintlab/data/sigkdd2021/PhaseII/testset', type=str, help='test data path')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu number')

parser.add_argument('--seq_len', type=int, default=100, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=50, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=25, help='prediction sequence length')

parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type, -1 = use target sequence as decoder input')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h', help='Time frequency')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', default=False,
                    help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--features', type=str, default='S',
                    help='forecasting task, options:[M, S, MS]; M: multivariate predict multivariate, '
                         'S:univariate predict univariate, MS: multivariate predict univariate')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
# parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser.add_argument('--device', type=str, default=device)

args = parser.parse_args()

def inverse(x, mini, maxi):
    output = mini + x * (maxi - mini)
    return output


def main():
    file_list = sorted(os.listdir(args.traindir))    # list all the file names under train_directory
    data_list = [file for file in file_list if file.endswith(".csv")]   # filter out file with .csv ending
    if args.dataset == 'e':
        data_list = ['e_DOM.csv', 'e_AEP.csv']
    elif args.dataset == 'stock':
        data_list = ['AMD.csv', 'NVDA.csv']

    for i in range(len(data_list)):
        type_dict = {
            'type1': [35, 14, 7],
            'type2': [70, 28, 14],
            'type3': [100, 50, 25],
            'type4': [200, 75, 35],
            'type5': [300, 150, 50]
        }

        for key in type_dict.keys():

            args.seq_len = type_dict[key][0]   # input sequence length of Informer encoder
            args.label_len = type_dict[key][1]  # start token length of Informer decoder
            args.pred_len = type_dict[key][2]  # prediction sequence length

            args.save_root = f'./exp_20200101_RevIN_freq_v2/{args.model}-{key}-{args.folder_name}/'

            train_dataset = dataloader.loader(args.traindir, data_list[i],
                                              seq_size=type_dict[key], loader_type='train', args=args)

            test_dataset = dataloader.loader(args.traindir, data_list[i],
                                             seq_size=type_dict[key], loader_type='test', args=args)

            save_path = os.path.join(args.save_root, data_list[i])

            args.save_path = save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            train_loader = DataLoader(train_dataset,
                                      shuffle=True,
                                      batch_size=args.batch_size,
                                      pin_memory=False)
            test_loader = DataLoader(test_dataset,
                                     shuffle=False,
                                     batch_size=args.batch_size,
                                     pin_memory=False)

            model_dict = {
                'informer': Informer,
                'informerstack': InformerStack,

            }
            if args.model == 'informer' or args.model == 'informerstack':
                e_layers = args.e_layers if args.model == 'informer' else args.s_layers  ################################
                net = model_dict[args.model](
                    args.enc_in,
                    args.dec_in,
                    args.c_out,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.factor,
                    args.d_model,
                    args.n_heads,
                    e_layers,  # self.args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.dropout,
                    args.attn,
                    args.embed,
                    args.freq,
                    args.activation,
                    args.output_attention,
                    args.distil,
                    args.mix
                ).float().to(device)

                # input_data = torch.randn(1, 100,1).cuda()
                # other_input_data = torch.randn(1, 300,1).cuda()

                # summaryx(net,input_data, other_input_data)
                # import ipdb; ipdb.set_trace()

            elif args.model == 'reformer':
                enc_bucket_size = args.seq_len // 2 if args.enc_bucket_size == 0 else args.enc_bucket_size
                dec_bucket_size = (
                                              args.label_len + args.pred_len) // 2 if args.enc_bucket_size == 0 else args.enc_bucket_size
                net = model_dict[args.model](dim=args.enc_in, seq_len=args.seq_len,
                                             label_len=args.label_len,
                                             pred_len=args.pred_len,
                                             enc_bucket_size=enc_bucket_size,  # default: maxlen 128 , bucket_size 64
                                             dec_bucket_size=dec_bucket_size,  # default: maxlen 128 , bucket_size 64
                                             enc_depth=args.enc_depth,
                                             dec_depth=args.dec_depth,
                                             enc_heads=args.enc_heads,  #default: 8
                                             dec_heads=args.dec_heads,  #default: 8
                                             enc_dim_head=args.enc_dim_head,
                                             dec_dim_head=args.dec_dim_head,
                                             enc_n_hashes=args.enc_n_hashes,
                                             dec_n_hashes=args.dec_n_hashes,
                                             enc_ff_chunks=args.enc_ff_chunks,
                                             dec_ff_chunks=args.dec_ff_chunks,
                                             enc_attn_chunks=args.enc_attn_chunks,
                                             dec_attn_chunks=args.dec_attn_chunks,
                                             enc_weight_tie=args.enc_weight_tie,
                                             dec_weight_tie=args.dec_weight_tie,
                                             enc_causal=args.enc_causal,
                                             dec_causal=args.dec_causal,
                                             enc_n_local_attn_heads=args.enc_n_local_attn_heads,
                                             dec_n_local_attn_heads=args.dec_n_local_attn_heads,
                                             enc_use_full_attn=args.enc_use_full_attn,
                                             dec_use_full_attn=args.dec_use_full_attn).to(device)

            # net = nn.DataParallel(net)
            total_params = sum(p.numel() for p in net.parameters())
            print(f"Number of parameters: {total_params}")
            criterion = nn.MSELoss().to(device)

            optimizer = optim.Adam(net.parameters(), lr=1e-3)

            train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
            test_logger = utils.Logger(os.path.join(save_path, 'test.log'))

            # Start Train
            for epoch in range(1, args.epochs + 1):
                epoch, loss = train(train_loader,
                                    net,
                                    criterion,
                                    optimizer,
                                    epoch,
                                    train_logger,
                                    args)
                epoch, tst_loss, preds, trues = test(test_loader, net, criterion, epoch, test_logger, args)

            pred, trues = evaluate(test_loader, net, criterion, args)

            torch.save(net.state_dict(),
                       os.path.join(save_path, f'model_{int(args.epochs)}.pth'))

            mse_losses = MSE(pred, trues)
            mae_losses = MAE(pred, trues)
            rmse_losses = RMSE(pred, trues)

            plt.figure(figsize=(64, 16))
            plt.plot(trues, color='blue', alpha=0.5, linewidth=3, label='input')
            plt.plot(pred, color='red', alpha=0.5, linewidth=3, label='output')
            plt.legend(['target', 'prediction'], prop={'size': 30})
            plt.savefig(f'{save_path}/{data_list[i]}_all.png')
            plt.close()

            # inverse scaling

            min_val, max_val = train_dataset.get_minmax()

            inverse_trues = inverse(trues, min_val, max_val)
            inverse_preds = inverse(pred, min_val, max_val)

            plt.figure(figsize=(64, 16))
            plt.plot(inverse_trues, color='blue', alpha=0.5, linewidth=3, label='input_raw')
            plt.plot(inverse_preds, color='red', alpha=0.5, linewidth=3, label='output_inverse')
            plt.legend(['target', 'prediction'], prop={'size': 30})
            plt.savefig(f'{save_path}/{data_list[i]}_all_inverse.png')
            plt.close()

            save_output = pd.DataFrame(
                {'data': trues, 'pred': pred, 'inverse_data': inverse_trues, 'inverse_pred': inverse_preds})

            save_output['MSE'] = mse_losses
            save_output['MAE'] = mae_losses
            save_output['RMSE'] = rmse_losses

            save_output.to_csv(f'{save_path}/output.csv')


if __name__ == "__main__":
    main()

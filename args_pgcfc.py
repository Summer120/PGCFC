import argparse
import configparser


def parse_pgcfc_args():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--dataset', default='PEMSD4', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
    parser.add_argument('--debug', default='False', type=eval)
    parser.add_argument('--model', default='PGCFC', type=str)
    parser.add_argument('--cuda', default=True, type=bool)

    args1 = parser.parse_args()

    config_file = './config_file/{}_{}.conf'.format(args1.dataset, args1.model)
    config = configparser.ConfigParser()
    config.read(config_file)

    parser.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    parser.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    parser.add_argument('--lag', default=config['data']['lag'], type=int)
    parser.add_argument('--horizon', default=config['data']['horizon'], type=int)
    parser.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    parser.add_argument('--tod', default=config['data']['tod'], type=eval)
    parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    parser.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    parser.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
    parser.add_argument('--steps_per_day', default=config['data']['steps_per_day'], type=int)
    parser.add_argument('--steps_per_week', default=config['data']['steps_per_week'], type=int)

    parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--time_dim', default=config['model']['time_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    parser.add_argument('--use_day', default=config['model']['use_day'], type=eval)
    parser.add_argument('--use_week', default=config['model']['use_week'], type=eval)

    parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    parser.add_argument('--seed', default=config['train']['seed'], type=int)
    parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser.add_argument('--weight_decay', default=config['train']['weight_decay'], type=float)
    parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=float)
    parser.add_argument('--lr_decay_step1', default=config['train']['lr_decay_step1'], type=str)
    parser.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    parser.add_argument('--teacher_forcing', default=False, type=bool)
    parser.add_argument('--real_value', default=config['train']['real_value'], type=eval, help='use real value for loss calculation')

    parser.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    parser.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)

    parser.add_argument('--log_dir', default='./', type=str)
    parser.add_argument('--log_step', default=config['log']['log_step'], type=int)
    parser.add_argument('--plot', default=config['log']['plot'], type=eval)

    return parser.parse_args()

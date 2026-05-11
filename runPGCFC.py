import os
import sys
import torch
import math
import copy
import time
import numpy as np
import pynvml
import torch.nn as nn
from datetime import datetime
from lib.logger import get_logger
from lib.metrics import MAE_torch, All_Metrics
from lib.TrainInits import init_seed, print_model_parameters
from lib.dataloader import get_dataloader
from model.PGCFC import PGCFC as Network
from args_pgcfc import parse_pgcfc_args
import warnings
warnings.filterwarnings('ignore')

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.val_per_epoch = len(val_loader) if val_loader is not None else 0
        
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_test_path = os.path.join(self.args.log_dir, 'best_test_model.pth')
        
        if not args.debug and not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        # --- 添加下面这行代码 ---
        self.logger.info('Running Script: {}'.format(os.path.basename(__file__)))
        self.logger.info(args)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.batches_seen = 0
        self.meminfo = 0

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                label = target[..., :self.args.output_dim].clone()
                target[...,:self.args.output_dim] = self.scaler.transform(target[...,:self.args.output_dim])
                output = self.model(data, target)
                
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                
                loss = self.loss(output.cuda(), label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('***********Val Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, val_loss, time.time() - epoch_time))
        return val_loss

    def test_epoch(self, epoch, test_dataloader):
        self.model.eval()
        total_test_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dataloader):
                label = target[..., :self.args.output_dim].clone()
                target[...,:self.args.output_dim] = self.scaler.transform(target[...,:self.args.output_dim])
                output = self.model(data, target)
                
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                
                loss = self.loss(output.cuda(), label)
                if not torch.isnan(loss):
                    total_test_loss += loss.item()
        
        test_loss = total_test_loss / len(test_dataloader)
        self.logger.info('**********test Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, test_loss, time.time() - epoch_time))
        return test_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        epoch_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.batches_seen += 1
            label = target[..., :self.args.output_dim].clone()
            target[..., :self.args.output_dim] = self.scaler.transform(target[..., :self.args.output_dim])
            self.optimizer.zero_grad()

            output = self.model(data, target, self.batches_seen)
            if self.args.real_value:
                output = self.scaler.inverse_transform(output)

            loss = self.loss(output.cuda(), label)
            loss.backward()

            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            self.optimizer.step()
            total_loss += loss.item()

            if (batch_idx+1) % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx+1, self.train_per_epoch, loss.item()))
        
        train_epoch_loss = total_loss / self.train_per_epoch
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.logger.info('********Train Epoch {}: averaged Loss: {:.6f}, GPU cost: {:.2f} GB, train time: {:.2f} s'.format(
            epoch, train_epoch_loss, (meminfo.used - self.meminfo.used) / 1024 ** 3, time.time() - epoch_time))

        if self.args.lr_decay:
            self.lr_scheduler.step()
        
        return train_epoch_loss

    def train(self):
        self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        best_model = None
        best_test_model = None
        not_improved_count = 0
        best_loss = float('inf')
        best_test_loss = float('inf')

        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss = self.train_epoch(epoch)

            val_dataloader = self.test_loader if self.val_loader is None else self.val_loader
            test_dataloader = self.test_loader

            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            test_epoch_loss = self.test_epoch(epoch, test_dataloader)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_model = copy.deepcopy(self.model.state_dict())
                self.logger.info('*********************************Current best model saved!')
            else:
                not_improved_count += 1

            if test_epoch_loss < best_test_loss:
                best_test_loss = test_epoch_loss
                best_test_model = copy.deepcopy(self.model.state_dict())

            if self.args.early_stop and not_improved_count == self.args.early_stop_patience:
                self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.args.early_stop_patience))
                break

        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            torch.save(best_test_model, self.best_test_path)
            self.logger.info("Saving current best model to " + self.best_test_path)

        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

        self.logger.info("This is best_test_model")
        self.model.load_state_dict(best_test_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path is not None:
            check_point = torch.load(path)
            model.load_state_dict(check_point['state_dict'])
            model.to(args.device)
        
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                label = target[..., :args.output_dim]
                output = model(data, target)
                y_true.append(label)
                y_pred.append(output)

        if args.real_value:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            y_true = torch.cat(y_true, dim=0)
        else:
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)

        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, corr = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%".format(
                t + 1, rmse, mae, mape*100))
        
        mae, rmse, mape, _, corr = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("test1 Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}%".format(
                    rmse, mae, mape*100))



file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        return MAE_torch(pred=preds, true=labels, mask_value=mask_value)
    return loss

args = parse_pgcfc_args()
print(args)

init_seed(args.seed)

if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

model = Network(args).to(args.device)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

train_loader, val_loader, test_loader, scaler = get_dataloader(
    args,
    normalizer=args.normalizer,
    tod=args.tod, dow=False,
    weather=False, single=False
)

if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError(f"Unsupported loss function: {args.loss_func}")

optimizer = torch.optim.Adam(
    params=model.parameters(), 
    lr=args.lr_init, 
    eps=1.0e-8,
    weight_decay=args.weight_decay, 
    amsgrad=False
)

lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in args.lr_decay_step1.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=lr_decay_steps,
        gamma=args.lr_decay_rate
    )

script_name = os.path.basename(__file__)
script_name_no_ext = os.path.splitext(script_name)[0]
current_time = f"{script_name_no_ext}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
current_dir = os.path.dirname(os.path.realpath(__file__))
args.log_dir = os.path.join(current_dir, 'experiments', args.dataset, current_time)

trainer = Trainer(
    model, loss, optimizer, 
    train_loader, val_loader, test_loader, 
    scaler, args, lr_scheduler=lr_scheduler
)

if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load(f'./pre-trained/{args.dataset}.pth'))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError(f"Unsupported mode: {args.mode}, only 'train' or 'test' is allowed")

import copy
import time
import torch
import os
import sys
import datetime
import numpy as np
import tqdm
import wandb

# from torch.utils.tensorboard import SummaryWriter

wandb_flags = True


class EarlyStopping:
    def __init__(self, out_dir, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.out_dir = out_dir

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, os.path.join(self.out_dir, 'checkpoint_max.pth'))
        self.val_loss_min = val_loss


def train_CNN(out_dir, min_test_loss, net, start_epoch, end_epoch, criterion,optimizer,lr_scheduler, train_data_loader, test_data_loader, args):
    if wandb_flags:
        wandb.init(project="fdtd_simulator", name=out_dir)
    
    # Early Stopping Initialization
    # early_stopping = EarlyStopping(out_dir=out_dir, patience = 10, verbose=True)

    min_test_loss = 100000
    if os.path.exists(out_dir):
        checkpoint = torch.load(out_dir + '/checkpoint_max.pth', map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        min_test_loss = checkpoint['min_test_loss']
        with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))
            args_txt.write('\n')
            args_txt.write(' '.join(sys.argv))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    # writer = SummaryWriter(out_dir, purge_step=start_epoch)


    initial_weights = copy.deepcopy(net.state_dict())
    weight_changes = []
    learning_rates = []
    for epoch in range(start_epoch, end_epoch):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        current_weights = copy.deepcopy(net.state_dict())
        weight_change = {}
        current_lr = lr_scheduler.get_last_lr()
        learning_rates.append(current_lr[0])


        # 参数变化曲线，可以训练情况
        for name, param in current_weights.items():
            if name in initial_weights:
                change = torch.sum(torch.abs(param - initial_weights[name]))
                weight_change[name] = change.item()
        weight_changes.append(weight_change)


        # 训练代码
        for batch_idx, data in enumerate(train_data_loader, 0):
            img, label = data
            #(B, H, W) to (B,C,H,W) C = 1 due to 01
            img = img.unsqueeze(1)

            img = img.to(args.device)
            label = label.to(args.device)
            optimizer.zero_grad()
            out_fr = net(img)
            loss = criterion(out_fr, label)
            loss.backward()
            optimizer.step()
            train_samples += args.b
            train_loss += loss.item()

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= len(train_data_loader)

        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        if wandb_flags:
            wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "epoch": epoch
            })

        lr_scheduler.step()

        # test mode
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_data_loader, 0):
                img, label = data
                img = img.unsqueeze(1)
                img = img.to(args.device)
                label = label.to(args.device)
                out_fr = net(img)
                loss = criterion(out_fr, label)
                test_samples += args.b
                test_loss += loss.item()


        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        train_loss /= len(train_data_loader)

        # writer.add_scalar('test_loss', test_loss, epoch)
        # writer.add_scalar('test_acc', test_acc, epoch)
        if wandb_flags:
            wandb.log({
            "test_loss": test_loss,
            "epoch": epoch
            })

        save_max = False
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            save_max = True


        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'min_test_loss': min_test_loss
        }

        # early stopping
        # early_stopping(test_loss, checkpoint)
        # if early_stopping.early_stop:
        #     print("Early stopping triggered")
        #     break

        # # sum of epoch is 100 
        if save_max :  
            print("time to save")
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
            


        print(args)
        print(out_dir)
        print(
            f'epoch = {epoch}, train_loss ={train_loss: .9f}, test_loss ={test_loss: 9f}, min_test_loss ={min_test_loss: .9f}, current_lr = {current_lr[0]: .6f}')
        print(f'train speed ={train_speed: .9f} images/s, test speed ={test_speed: .9f} images/s')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        # print("save or not",save_max)
        
    # save the last checkpoint
    torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
    np.save(out_dir + '/weight_changes.npy',  np.array(weight_changes))
    np.save(out_dir +'/learning_rates.npy',  np.array(learning_rates))
    return  weight_changes, learning_rates, net

def exponential_decay(epoch, initial_ratio, decay_rate):
    return initial_ratio * np.exp(-decay_rate * epoch)

def train_CRNNAG(out_dir, min_test_loss, net, start_epoch, end_epoch, criterion,optimizer,lr_scheduler, train_data_loader, test_data_loader, args, teacher_force_ratio = 0.5):
    if wandb_flags:
        wandb.init(project="fdtd_simulator", name=out_dir)
    
    # Early Stopping Initialization
    # early_stopping = EarlyStopping(out_dir=out_dir, patience = 15, verbose=True)


    train_loss_lst = []
    test_loss_lst = []
    best_valid_loss = float("inf")
    initial_teacher_forcing_ratio = teacher_force_ratio
    decay_rate = 0.02

    min_test_loss = 100000
    if os.path.exists(out_dir):
        checkpoint = torch.load(out_dir + '/checkpoint_max.pth', map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        min_test_loss = checkpoint['min_test_loss']
        with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))
            args_txt.write('\n')
            args_txt.write(' '.join(sys.argv))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    # writer = SummaryWriter(out_dir, purge_step=start_epoch)


    initial_weights = copy.deepcopy(net.state_dict())
    weight_changes = []
    learning_rates = []
    for epoch in range(start_epoch, end_epoch):
        teacher_forcing_ratio_per = exponential_decay(epoch, initial_teacher_forcing_ratio, decay_rate)
        print("teacher force ratio:",teacher_forcing_ratio_per)
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        current_weights = copy.deepcopy(net.state_dict())
        weight_change = {}
        current_lr = lr_scheduler.get_last_lr()
        learning_rates.append(current_lr[0])


        # 参数变化曲线，可以训练情况
        for name, param in current_weights.items():
            if name in initial_weights:
                change = torch.sum(torch.abs(param - initial_weights[name]))
                weight_change[name] = change.item()
        weight_changes.append(weight_change)


        # 训练代码
        for batch_idx, data in enumerate(train_data_loader, 0):

            img, label = data
            #(B, H, W) to (B,C,H,W) C = 1 due to 01
            img = img.unsqueeze(1)

            img = img.to(args.device)
            label = label.to(args.device)

            # 梯度裁剪（在optimizer.step()前添加）
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            #teacher_force_ratio 是 训练时用label 的概率
            out_fr = net(img, label,teacher_forcing_ratio_per)
            loss = criterion(out_fr, label)
            loss.backward()
            optimizer.step()
            train_samples += args.b
            train_loss += loss.item()

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= len(train_data_loader)

        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        if wandb_flags:
            wandb.log({
            "train_loss": train_loss,
            "epoch": epoch
            })

        lr_scheduler.step()

        # test mode
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_data_loader, 0):
                img, label = data
                img = img.unsqueeze(1)
                img = img.to(args.device)
                label = label.to(args.device)
                out_fr = net(img, label, 0)
                loss = criterion(out_fr, label)
                test_samples += args.b
                test_loss += loss.item()


        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        train_loss /= len(train_data_loader)
        
        # writer.add_scalar('test_loss', test_loss, epoch)
        # writer.add_scalar('test_acc', test_acc, epoch)
        if wandb_flags:
            wandb.log({
            "test_loss": test_loss,
            "epoch": epoch
            })

        save_max = False
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            save_max = True
        

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'min_test_loss': min_test_loss
        }

        # # Early Stopping
        # early_stopping(test_loss, checkpoint)
        # if early_stopping.early_stop :
        #     print("Early stopping triggered")
        #     break

        # # sum of epoch is 100 
        if save_max and epoch > 10:  
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))


        print(args)
        print(out_dir)
        print(
            f'epoch = {epoch}, train_loss ={train_loss: .9f}, test_loss ={test_loss: 9f}, min_test_loss ={min_test_loss: .9f}, current_lr = {current_lr[0]: .6f}')
        print(f'train speed ={train_speed: .9f} images/s, test speed ={test_speed: .9f} images/s')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
 
    # save the last checkpoint
    torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
    np.save(out_dir + '/weight_changes.npy',  np.array(weight_changes))
    np.save(out_dir +'/learning_rates.npy',  np.array(learning_rates))
    return  weight_changes, learning_rates, net

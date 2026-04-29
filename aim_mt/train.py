import pandas as pd
import os
import cv2
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
from torch import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

import shutil
from model import aim_mt
from data import Go2_Data
from config import GlobalConfig
from torch.utils.tensorboard import SummaryWriter
# import random
# random.seed(0)
# torch.manual_seed(0)

"""
#buat ngecek GT SEG aja
def check_gt_seg(config, gt_seg):
    gt_seg = gt_seg.cpu().detach().numpy()

    #buat array untuk nyimpan out gambar
    imgx = np.zeros((gt_seg.shape[2], gt_seg.shape[3], 3))
    #ambil tensor segmentationnya
    inx = np.argmax(gt_seg[0], axis=0)
    for cmap in config.SEG_CLASSES['colors']:
        cmap_id = config.SEG_CLASSES['colors'].index(cmap)
        imgx[np.where(inx == cmap_id)] = cmap
    
    #GANTI ORDER BGR KE RGB, SWAP!
    imgx = swap_RGB2BGR(imgx)
    cv2.imwrite(config.logdir+"/check_gt_seg.png", imgx) #cetak gt segmentation
"""

#Class untuk penyimpanan dan perhitungan update loss
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    #update kalkulasi
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#loss depth est
def l1l2loss (Yp, Yt):
    mae = F.l1_loss(Yp, Yt)
    mse = F.mse_loss(Yp, Yt)
    l1l2 = mae+mse
    return l1l2

#Class NN Module untuk Perhitungan BCE Dice Loss
def BCEDice(Yp, Yt, smooth=1e-7):
    #.view(-1) artinya matrix tensornya di flatten kan dulu
    Yp = Yp.view(-1)
    Yt = Yt.view(-1)
    #hitung BCE
    bce = F.binary_cross_entropy(Yp, Yt, reduction='mean')
    #hitung dice loss
    intersection = (Yp * Yt).sum() #irisan
    #rumus DICE
    dice_loss = 1 - ((2. * intersection + smooth) / (Yp.sum() + Yt.sum() + smooth))
    #kalkulasi lossnya
    bce_dice_loss = bce + dice_loss
    return bce_dice_loss


# def weighted_bce(Yp, Yt, weights=[1, 1.5]): #weights untuk class 0 dan class 1
#     Yt = Yt.view(-1)
#     Yp = torch.clamp(Yp.view(-1), min=1e-7, max=1-1e-7)
#     loss = -1 * torch.mean(weights[1]*Yt*torch.log(Yp) + weights[0]*(1-Yt)*torch.log(1-Yp))
#     return loss


#fungsi renormalize loss weights seperti di paper gradnorm
def renormalize_params_lw(current_lw, config):
    #detach dulu paramsnya dari torch, pindah ke CPU
    lw = np.array([tens.cpu().detach().numpy() for tens in current_lw])
    lws = np.array([lw[i][0] for i in range(len(lw))])
    #fungsi renormalize untuk algoritma 1 di papaer gradnorm
    coef = np.array(config.loss_weights).sum()/lws.sum()
    new_lws = [coef*lwx for lwx in lws]
    #buat torch float tensor lagi dan masukkan ke cuda memory
    normalized_lws = [torch.cuda.FloatTensor([lw]).clone().detach().requires_grad_(True) for lw in new_lws]
    return normalized_lws

#FUNGSI TRAINING
def train(data_loader, model, config, writer, cur_epoch, optimizer, params_lw, optimizer_lw):
    #buat variabel untuk menyimpan kalkulasi loss, dan iou
    score = {'total_loss': AverageMeter(),
            'ss_loss': AverageMeter(),
            'de_loss': AverageMeter(),
            'wp_loss': AverageMeter()}
    
    #masuk ke mode training, pytorch
    model.train()

    #visualisasi progress training dengan tqdm
    prog_bar = tqdm(total=len(data_loader))

    #training....
    total_batch = len(data_loader)
    batch_ke = 0
    for data in data_loader:
        cur_step = cur_epoch*total_batch + batch_ke

        #load IO dan pindah ke GPU
        rgbs = []
        segs = []
        deps = []
        #d_cld_xs = []
        #d_cld_zs = []
        for i in range(0, config.seq_len): #append data untuk input sequence
            rgbs.append(data['rgbs'][i].to(config.gpu_device, dtype=config.dtype))
            segs.append(data['rgb_segs'][i].to(config.gpu_device, dtype=config.dtype))
            deps.append(data['depths'][i].to(config.gpu_device, dtype=config.dtype))
            # check_gt_seg(config, segs[-1])
            #d_cld_xs.append(data['d_cld_xs'][i].to(config.gpu_device, dtype=config.dtype))
            #d_cld_zs.append(data['d_cld_zs'][i].to(config.gpu_device, dtype=config.dtype))
        rp1 = torch.stack(data['rp1'], dim=1).to(config.gpu_device, dtype=config.dtype)
        rp2 = torch.stack(data['rp2'], dim=1).to(config.gpu_device, dtype=config.dtype)
        #gt_velocity = torch.stack([data['velo']], dim=1).to(config.gpu_device, dtype=config.dtype)
        gt_waypoints = [torch.stack(data['waypoints'][j], dim=1).to(config.gpu_device, dtype=config.dtype) for j in range(0, config.pred_len)]
        gt_waypoints = torch.stack(gt_waypoints, dim=1).to(config.gpu_device, dtype=config.dtype)
        #gt_robot_ctrl = torch.stack(data['robot_ctrl'], dim=1).to(config.gpu_device, dtype=config.dtype)

        #forward pass
        pred_segs, pred_deps, pred_wp, = model(rgbs, rp1, rp2)#, seg_fronts[-1])
        # check_gt_seg(config, sdcs[-1])

        #compute loss
        loss_seg = 0
        loss_dep = 0
        for i in range(0, config.seq_len):
            loss_seg = loss_seg + BCEDice(pred_segs[i], segs[i])
            loss_dep = loss_dep + l1l2loss(pred_deps[i], deps[i])
        loss_seg = loss_seg / config.seq_len #dirata-rata
        loss_dep = loss_dep / config.seq_len #dirata-rata
        loss_wp = l1l2loss(pred_wp, gt_waypoints)
        total_loss = params_lw[0]*loss_seg + params_lw[1]*loss_dep + params_lw[2]*loss_wp

        #backpro, kalkulasi gradient, dan optimasi
        optimizer.zero_grad()

        if batch_ke == 0: #batch pertama, hitung loss awal
            total_loss.backward() #ga usah retain graph
            #ambil loss pertama
            loss_seg_0 = torch.clone(loss_seg)
            loss_dep_0 = torch.clone(loss_dep)
            loss_wp_0 = torch.clone(loss_wp)

        elif 0 < batch_ke < total_batch-1:
            total_loss.backward() #ga usah retain graph

        elif batch_ke == total_batch-1: #berarti batch terakhir, compute update loss weights
            if config.MGN:
                optimizer_lw.zero_grad()
                total_loss.backward(retain_graph=True) #backpro, hitung gradient, retain graph karena graphnya masih dipakai perhitungan
                #ambil nilai gradient dari layer pertama pada masing2 task-specified decoder dan komputasi gradient dari output layer sampai ke bottle neck saja
                params = list(filter(lambda p: p.requires_grad, model.parameters()))
                G0R = torch.autograd.grad(loss_seg, params[config.bottleneck[0]], retain_graph=True, create_graph=True)
                G0 = torch.norm(G0R[0], keepdim=True)
                G1R = torch.autograd.grad(loss_dep, params[config.bottleneck[0]], retain_graph=True, create_graph=True)
                G1 = torch.norm(G1R[0], keepdim=True)
                G2R = torch.autograd.grad(loss_wp, params[config.bottleneck[1]], retain_graph=True, create_graph=True)
                G2 = torch.norm(G2R[0], keepdim=True)
                #dan rata2
                G_avg = (G0+G1+G2) / len(config.loss_weights)

                #hitung relative lossnya
                loss_seg_hat = loss_seg / loss_seg_0
                loss_dep_hat = loss_dep / loss_dep_0
                loss_wp_hat = loss_wp / loss_wp_0
                #dan rata2
                loss_hat_avg = (loss_seg_hat + loss_dep_hat + loss_wp_hat) / len(config.loss_weights)

                #hitung r_i_(t) relative inverse training rate untuk setiap task 
                inv_rate_ss = loss_seg_hat / loss_hat_avg
                inv_rate_de = loss_dep_hat / loss_hat_avg
                inv_rate_wp = loss_wp_hat / loss_hat_avg

                #hitung constant target grad
                C0 = (G_avg*inv_rate_ss).detach()**config.lw_alpha
                C1 = (G_avg*inv_rate_de).detach()**config.lw_alpha
                C2 = (G_avg*inv_rate_wp).detach()**config.lw_alpha

                #HITUNG TOTAL LGRAD
                Lgrad = F.l1_loss(G0, C0) + F.l1_loss(G1, C1) + F.l1_loss(G2, C2)

                #hitung gradient loss sesuai Eq. 2 di GradNorm paper
                # optimizer_lw.zero_grad()
                Lgrad.backward()
                #update loss weights
                optimizer_lw.step() 

                #ambil lgrad untuk disimpan nantinya
                lgrad = Lgrad.item()
                new_param_lw = optimizer_lw.param_groups[0]['params']
                # print(new_param_lw)
            else:
                total_loss.backward()
                lgrad = 0
                new_param_lw = 1
            
        optimizer.step() #dan update bobot2 pada network model

        #hitung rata-rata (avg) loss, dan metric untuk batch-batch yang telah diproses
        score['total_loss'].update(total_loss.item())
        score['ss_loss'].update(loss_seg.item()) 
        score['de_loss'].update(loss_dep.item()) 
        score['wp_loss'].update(loss_wp.item())

        #update visualisasi progress bar
        postfix = OrderedDict([('t_total_l', score['total_loss'].avg),
                            ('t_ss_l', score['ss_loss'].avg),
                            ('t_de_l', score['de_loss'].avg),
                            ('t_wp_l', score['wp_loss'].avg)])
        
        #tambahkan ke summary writer
        writer.add_scalar('t_total_l', total_loss.item(), cur_step)
        writer.add_scalar('t_ss_l', loss_seg.item(), cur_step)
        writer.add_scalar('t_de_l', loss_dep.item(), cur_step)
        writer.add_scalar('t_wp_l', loss_wp.item(), cur_step)

        prog_bar.set_postfix(postfix)
        prog_bar.update(1)
        batch_ke += 1
    prog_bar.close()    

    #return value
    return postfix, new_param_lw, lgrad


#FUNGSI VALIDATION
def validate(data_loader, model, config, writer, cur_epoch):
    #buat variabel untuk menyimpan kalkulasi loss, dan iou
    score = {'total_loss': AverageMeter(),
            'ss_loss': AverageMeter(),
            'de_loss': AverageMeter(),
            'wp_loss': AverageMeter()}
            
    #masuk ke mode eval, pytorch
    model.eval()

    with torch.no_grad():
        #visualisasi progress validasi dengan tqdm
        prog_bar = tqdm(total=len(data_loader))

        #validasi....
        total_batch = len(data_loader)
        batch_ke = 0
        for data in data_loader:
            cur_step = cur_epoch*total_batch + batch_ke

            #load IO dan pindah ke GPU
            rgbs = []
            segs = []
            deps = []
            d_cld_xs = []
            d_cld_zs = []
            for i in range(0, config.seq_len): #append data untuk input sequence
                rgbs.append(data['rgbs'][i].to(config.gpu_device, dtype=config.dtype))
                segs.append(data['rgb_segs'][i].to(config.gpu_device, dtype=config.dtype))
                deps.append(data['depths'][i].to(config.gpu_device, dtype=config.dtype))
                # check_gt_seg(config, segs[-1])
                d_cld_xs.append(data['d_cld_xs'][i].to(config.gpu_device, dtype=config.dtype))
                d_cld_zs.append(data['d_cld_zs'][i].to(config.gpu_device, dtype=config.dtype))
            rp1 = torch.stack(data['rp1'], dim=1).to(config.gpu_device, dtype=config.dtype)
            rp2 = torch.stack(data['rp2'], dim=1).to(config.gpu_device, dtype=config.dtype)
            #gt_velocity = torch.stack([data['velo']], dim=1).to(config.gpu_device, dtype=config.dtype)
            gt_waypoints = [torch.stack(data['waypoints'][j], dim=1).to(config.gpu_device, dtype=config.dtype) for j in range(0, config.pred_len)]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(config.gpu_device, dtype=config.dtype)
            #gt_robot_ctrl = torch.stack(data['robot_ctrl'], dim=1).to(config.gpu_device, dtype=config.dtype)
            #print(data['filename'])

            #forward pass
            pred_segs, pred_deps, pred_wp, = model(rgbs, rp1, rp2)#, seg_fronts[-1])
        
            #compute loss
            loss_seg = 0
            loss_dep = 0
            for i in range(0, config.seq_len):
                loss_seg = loss_seg + BCEDice(pred_segs[i], segs[i])
                loss_dep = loss_dep + l1l2loss(pred_deps[i], deps[i])
            loss_seg = loss_seg / config.seq_len #dirata-rata
            loss_dep = loss_dep / config.seq_len #dirata-rata
            loss_wp = l1l2loss(pred_wp, gt_waypoints)
            total_loss = loss_seg + loss_dep + loss_wp

            #hitung rata-rata (avg) loss, dan metric untuk batch-batch yang telah diproses
            score['total_loss'].update(total_loss.item())
            score['ss_loss'].update(loss_seg.item()) 
            score['de_loss'].update(loss_dep.item()) 
            score['wp_loss'].update(loss_wp.item())

            #update visualisasi progress bar
            postfix = OrderedDict([('v_total_l', score['total_loss'].avg),
                                ('v_ss_l', score['ss_loss'].avg),
                                ('v_de_l', score['de_loss'].avg),
                                ('v_wp_l', score['wp_loss'].avg)])
            
            #tambahkan ke summary writer
            writer.add_scalar('v_total_l', total_loss.item(), cur_step)
            writer.add_scalar('v_ss_l', loss_seg.item(), cur_step)
            writer.add_scalar('v_de_l', loss_dep.item(), cur_step)
            writer.add_scalar('v_wp_l', loss_wp.item(), cur_step)

            prog_bar.set_postfix(postfix)
            prog_bar.update(1)
            batch_ke += 1
        prog_bar.close()    

    #return value
    return postfix


#MAIN FUNCTION
def main():
    # Load config
    config = GlobalConfig()
    

    #SET GPU YANG AKTIF
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id#visible_gpu #"0" "1" "0,1"

    #IMPORT MODEL UNTUK DITRAIN
    print("IMPORT ARSITEKTUR DL DAN COMPILE")
    model = aim_mt(config).to(config.gpu_device, dtype=config.dtype)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)

    #KONFIGURASI OPTIMIZER
    # optima = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    optima = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optima, mode='min', factor=0.5, patience=config.lr_patience, min_lr=1e-7)

    #BUAT DATA BATCH
    train_set = Go2_Data(data_root=config.train_dir, config=config)
    val_set = Go2_Data(data_root=config.val_dir, config=config)
    # train_set = WHILL_Data(root=config.train_data, config=config)
    # val_set = WHILL_Data(root=config.val_data, config=config)
    # print(len(train_set))
    """"""
    if len(train_set)%config.batch_size == 1:
        drop_last = True #supaya ga mengacaukan MGN #drop last perlu untuk MGN
    else: #selain 1 bisa
        drop_last = False
    dataloader_train = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last) 
    dataloader_val = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)#, drop_last=True)
    # print(len(dataloader_train))
    
    #cek retrain atau tidak
    if not os.path.exists(config.logdir+"/trainval_log.csv"):
        print('TRAIN from the beginning!!!!!!!!!!!!!!!!')
        os.makedirs(config.logdir, exist_ok=True)
        print('Created dir:', config.logdir)
        #optimizer lw
        params_lw = [torch.cuda.FloatTensor([config.loss_weights[i]]).clone().detach().requires_grad_(True) for i in range(len(config.loss_weights))]
        optima_lw = optim.SGD(params_lw, lr=config.lr)
        #set nilai awal
        curr_ep = 0
        lowest_score = float('inf')
        stop_count = config.init_stop_counter
    else:
        print('Continue training!!!!!!!!!!!!!!!!')
        print('Loading checkpoint from ' + config.logdir)
        #baca log history training sebelumnya
        log_trainval = pd.read_csv(config.logdir+"/trainval_log.csv")
        # replace variable2 ini
        # print(log_trainval['epoch'][-1:])
        curr_ep = int(log_trainval['epoch'][-1:]) + 1
        lowest_score = float(np.min(log_trainval['val_loss']))
        stop_count = int(log_trainval['stop_counter'][-1:])
        # Load checkpoint
        model.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_model.pth')))
        optima.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_optim.pth')))

        #set optima lw baru
        latest_lw = [float(log_trainval['lw_ss'][-1:]), float(log_trainval['lw_de'][-1:]), float(log_trainval['lw_wp'][-1:])]
        params_lw = [torch.cuda.FloatTensor([latest_lw[i]]).clone().detach().requires_grad_(True) for i in range(len(latest_lw))]
        optima_lw = optim.SGD(params_lw, lr=float(log_trainval['lrate'][-1:]))
        # optima_lw.param_groups[0]['lr'] = optima.param_groups[0]['lr'] # lr disamakan
        # optima_lw.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_optim_lw.pth')))
        #update direktori dan buat tempat penyimpanan baru
        config.logdir += "/retrain"
        os.makedirs(config.logdir, exist_ok=True)
        print('Created new retrain dir:', config.logdir)
    
    #copykan config file
    shutil.copyfile('config.py', config.logdir+'/config.py')

    #buat dictionary log untuk menyimpan training log di CSV
    log = OrderedDict([
            ('epoch', []),
            ('best_model', []),
            ('val_loss', []),
            ('val_ss_loss', []),
            ('val_de_loss', []),
            ('val_wp_loss', []),
            ('train_loss', []), 
            ('train_ss_loss', []),
            ('train_de_loss', []),
            ('train_wp_loss', []),
            ('lrate', []),
            ('stop_counter', []), 
            ('lgrad_loss', []),
            ('lw_ss', []),
            ('lw_de', []),
            ('lw_wp', []),
            ('elapsed_time', []),
        ])
    writer = SummaryWriter(log_dir=config.logdir)
    
    #proses iterasi tiap epoch
    epoch = curr_ep
    while True:
        print("Epoch: {:05d}------------------------------------------------".format(epoch))
        #cetak lr dan lw
        if config.MGN:
            curr_lw = optima_lw.param_groups[0]['params']
            lw = np.array([tens.cpu().detach().numpy() for tens in curr_lw])
            lws = np.array([lw[i][0] for i in range(len(lw))])
            print("current loss weights: ", lws)    
        else:
            curr_lw = config.loss_weights
            lws = config.loss_weights
            print("current loss weights: ", config.loss_weights)
        print("current lr untuk training: ", optima.param_groups[0]['lr'])

        #training validation
        start_time = time.time() #waktu mulai
        train_log, new_params_lw, lgrad = train(dataloader_train, model, config, writer, epoch, optima, curr_lw, optima_lw)
        val_log = validate(dataloader_val, model, config, writer, epoch)
        if config.MGN:
            #update params lw yang sudah di renormalisasi ke optima_lw
            optima_lw.param_groups[0]['params'] = renormalize_params_lw(new_params_lw, config) #harus diclone supaya benar2 terpisah
            print("total loss gradient: "+str(lgrad))
        #update learning rate untuk training process
        scheduler.step(val_log['v_total_l']) #parameter acuan reduce LR adalah val_total_metric
        optima_lw.param_groups[0]['lr'] = optima.param_groups[0]['lr'] #update lr disamakan
        elapsed_time = time.time() - start_time #hitung elapsedtime

        #simpan history training ke file csv
        log['epoch'].append(epoch)
        log['lrate'].append(optima.param_groups[0]['lr'])
        log['train_loss'].append(train_log['t_total_l'])
        log['val_loss'].append(val_log['v_total_l'])
        log['train_ss_loss'].append(train_log['t_ss_l'])
        log['val_ss_loss'].append(val_log['v_ss_l'])
        log['train_de_loss'].append(train_log['t_de_l'])
        log['val_de_loss'].append(val_log['v_de_l'])
        log['train_wp_loss'].append(train_log['t_wp_l'])
        log['val_wp_loss'].append(val_log['v_wp_l'])
        log['lgrad_loss'].append(lgrad)
        log['lw_ss'].append(lws[0])
        log['lw_de'].append(lws[1])
        log['lw_wp'].append(lws[2])
        log['elapsed_time'].append(elapsed_time)
        print('| t_total_l: %.4f | t_ss_l: %.4f | t_de_l: %.4f | t_wp_l: %.4f |' % (train_log['t_total_l'], train_log['t_ss_l'], train_log['t_de_l'], train_log['t_wp_l']))
        print('| v_total_l: %.4f | v_ss_l: %.4f | v_de_l: %.4f | v_wp_l: %.4f |' % (val_log['v_total_l'], val_log['v_ss_l'], val_log['v_de_l'], val_log['v_wp_l']))
        print('elapsed time: %.4f sec' % (elapsed_time))
        
        #save recent model dan optimizernya
        torch.save(model.state_dict(), os.path.join(config.logdir, 'recent_model.pth'))
        torch.save(optima.state_dict(), os.path.join(config.logdir, 'recent_optim.pth'))
        # torch.save(optima_lw.state_dict(), os.path.join(config.logdir, 'recent_optim_lw.pth'))

        #save model best only
        if val_log['v_total_l'] < lowest_score:
            print("v_total_l: %.4f < lowest sebelumnya: %.4f" % (val_log['v_total_l'], lowest_score))
            print("model terbaik disave!")
            torch.save(model.state_dict(), os.path.join(config.logdir, 'best_model.pth'))
            torch.save(optima.state_dict(), os.path.join(config.logdir, 'best_optim.pth'))
            # torch.save(optima_lw.state_dict(), os.path.join(config.logdir, 'best_optim_lw.pth'))
            #v_total_l sekarang menjadi lowest_score
            lowest_score = val_log['v_total_l']
            #reset stop counter
            stop_count = config.init_stop_counter
            print("stop counter direset ke: ", stop_count)
            #catat sebagai best model
            log['best_model'].append("BEST")
        else:
            print("v_total_l: %.4f >= lowest sebelumnya: %.4f" % (val_log['v_total_l'], lowest_score))
            print("model tidak disave!")
            stop_count -= 1
            print("stop counter : ", stop_count)
            log['best_model'].append("")

        #update stop counter
        log['stop_counter'].append(stop_count)
        #paste ke csv file
        pd.DataFrame(log).to_csv(os.path.join(config.logdir, 'trainval_log.csv'), index=False)

        #kosongkan cuda chace
        torch.cuda.empty_cache()
        epoch += 1

        # early stopping jika stop counter sudah mencapai 0 dan early stop true
        if stop_count==0:
            print("TRAINING BERHENTI KARENA TIDAK ADA PENURUNAN TOTAL LOSS DALAM %d EPOCH TERAKHIR" % (config.init_stop_counter))
            break #loop
        

#RUN PROGRAM
if __name__ == "__main__":
    main()



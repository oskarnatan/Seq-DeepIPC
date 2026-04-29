import pandas as pd
import os
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
import cv2
from torch import torch
import yaml

from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from model import huang
from data import Go2_Data, colorize_seg, colorize_logdepth, colorize_depth
from log.huang_seq1_pred5.config import GlobalConfig #pakai config.py yang dicopykan ke log
# import random
# random.seed(0)
# torch.manual_seed(0)


#Class untuk penyimpanan dan perhitungan update metric
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


def IOU(Yp, Yt):
    #.view(-1) artinya matrix tensornya di flatten kan dulu
    output = Yp.view(-1) > 0.5 #maksudnya yang lebih dari 0.5 adalah true
    target = Yt.view(-1) > 0.5 #dan yang kurang dari 0.5 adalah false
    intersection = (output & target).sum() #irisan
    union = (output | target).sum() #union
    #rumus IoU
    iou = intersection / union
    return iou


"""
def get_wprp_frame(config, wp, rp1, rp2, velo, ctrl, bearing, seq, latlon):
    meta_output = dict()
    frame_h, frame_w = config.bev_h, config.bev_w
    area = config.coverage_area

    wp_frame = []
    #proses wp
    """"""
    for i in range(0, config.pred_len):
        x_frame = np.clip(int((wp[i][0]+area)*(frame_w-1)/(2*area)), 0, frame_w-1)#constrain
        y_frame = np.clip(int((wp[i][1]*(1-frame_h)/area) + (frame_h-1)), 0, frame_h-1)#constrain
        wp_frame.append(np.array([x_frame, y_frame]))
    
    #proses juga untuk next route
    rp1_x_frame = np.clip(int((rp1[0]+area)*(frame_w-1)/(2*area)), 0, frame_w-1)#constrain
    rp1_y_frame = np.clip(int((rp1[1]*(1-frame_h)/area) + (frame_h-1)), 0, frame_h-1)#constrain
    rp1_frame = np.array([rp1_x_frame, rp1_y_frame])

    rp2_x_frame = np.clip(int((rp2[0]+area)*(frame_w-1)/(2*area)), 0, frame_w-1)#constrain
    rp2_y_frame = np.clip(int((rp2[1]*(1-frame_h)/area) + (frame_h-1)), 0, frame_h-1)#constrain
    rp2_frame = np.array([rp2_x_frame, rp2_y_frame])

    meta_output['wp_frame'] = np.array(wp_frame).tolist()
    meta_output['rp1_frame'] = np.array(rp1_frame).tolist()
    meta_output['rp2_frame'] = np.array(rp2_frame).tolist()
    meta_output['wp'] = np.array(wp).tolist()
    meta_output['rp1'] = np.array(rp1).tolist()
    meta_output['rp2'] = np.array(rp2).tolist()
    meta_output['velo'] = np.array(velo).tolist()
    meta_output['ctrl'] = np.array(ctrl).tolist()
    meta_output['bearing'] = bearing
    meta_output['seq'] = int(seq)
    meta_output['latlon'] = latlon

    return meta_output


def save_out(config, seg, sdc, save_dir, route, filename, metax):
    seg = seg.cpu().detach().numpy()
    sdc = sdc.cpu().detach().numpy()

    #buat array untuk nyimpan out gambar
    imgx = np.zeros((seg.shape[2], seg.shape[3], 3))
    imgx2 = np.zeros((sdc.shape[2], sdc.shape[3], 3))
    #ambil tensor output segmentationnya
    pred_seg = seg[0]
    pred_sdc = sdc[0]
    inx = np.argmax(pred_seg, axis=0)
    inx2 = np.argmax(pred_sdc, axis=0)
    for cmap in config.cityscapes_palette:
        cmap_id = config.cityscapes_palette.index(cmap)
        imgx[np.where(inx == cmap_id)] = cmap
        imgx2[np.where(inx2 == cmap_id)] = cmap
    
    #GANTI ORDER BGR KE RGB, SWAP!
    imgx = imgx[:, :, [2, 1, 0]]
    imgx2 = imgx2[:, :, [2, 1, 0]]

    #save!
    save_dir_seg = save_dir+'/outputs/'+route+'/pred_seg/'
    save_dir_sdc = save_dir+'/outputs/'+route+'/sdc2/'
    os.makedirs(save_dir_seg, exist_ok=True)
    os.makedirs(save_dir_sdc, exist_ok=True)
    cv2.imwrite(save_dir_seg+filename, imgx) #cetak predicted segmentation
    cv2.imwrite(save_dir_sdc+filename, imgx2) #cetak predicted segmentation

    #buat sdc plot wprp dan metadata lainnya
    save_dir_sdc_wprp = save_dir+'/outputs/'+route+'/sdc2_wprp/'
    os.makedirs(save_dir_sdc_wprp, exist_ok=True)

    hhh = int(config.crop_roi[0]/config.scale)
    imgx3 = imgx2.copy()
    #plot rp
    imgx3 = cv2.circle(imgx3, (metax['rp1_frame'][0], metax['rp1_frame'][1]), radius=3, color=(255, 255, 255), thickness=1)
    imgx3 = cv2.circle(imgx3, (metax['rp2_frame'][0], metax['rp2_frame'][1]), radius=3, color=(255, 255, 255), thickness=1)
    for k in range(config.pred_len): #plot wp
        imgx3 = cv2.circle(imgx3, (metax['wp_frame'][k][0], metax['wp_frame'][k][1]), radius=1, color=(255, 255, 255), thickness=-1)
    cv2.putText(imgx3, "Sequence: "+str(metax['seq']), (10, hhh-80), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1, 2)
    cv2.putText(imgx3, "Steering: "+str(np.round(metax['str'], decimals=5)), (10, hhh-70), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1, 2)
    cv2.putText(imgx3, "Throttle: "+str(np.round(metax['thrt'], decimals=5)), (10, hhh-60), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1, 2)
    cv2.putText(imgx3, "Velocity: "+str(np.round(metax['velo'], decimals=5)), (10, hhh-50), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1, 2)
    cv2.putText(imgx3, "RP 1: "+str(np.round(metax['rp1'], decimals=5)), (10, hhh-40), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1, 2)
    cv2.putText(imgx3, "RP 2: "+str(np.round(metax['rp2'], decimals=5)), (10, hhh-30), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1, 2)
    cv2.putText(imgx3, "Bearing: "+str(np.round(metax['bearing'], decimals=5)), (10, hhh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1, 2)
    cv2.putText(imgx3, "Lat-Lon: "+str(np.round(metax['latlon'], decimals=5)), (10, hhh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1, 2)

    cv2.imwrite(save_dir_sdc_wprp+filename, imgx3) #cetak predicted segmentation

    #save juga yamlnya,
    save_dir_meta = save_dir+'/outputs/'+route+'/metax/'
    os.makedirs(save_dir_meta, exist_ok=True)
    with open(save_dir_meta+filename[:-3]+"yml", 'w') as dict_file:
        yaml.dump(metax, dict_file)
"""


#FUNGSI test
def test(data_loader, model, config):
    #buat variabel untuk menyimpan kalkulasi metric, dan iou
    score = {'total_metric': AverageMeter(),
            'ss_metric': AverageMeter(),
            'ctrl_metric': AverageMeter()}

    #buat dictionary log untuk menyimpan training log di CSV
    log = OrderedDict([
        ('batch', []),
        ('test_metric', []),
        ('test_ss_metric', []),
        ('test_ctrl_metric', []),
        ('elapsed_time', []),
    ])

    #buat save direktori
    save_dir = config.logdir + "/offline_test/" 
    os.makedirs(save_dir, exist_ok=True)
            
    #masuk ke mode eval, pytorch
    model.eval()

    with torch.no_grad():
        #visualisasi progress validasi dengan tqdm
        prog_bar = tqdm(total=len(data_loader))

        #validasi....
        batch_ke = 1
        for data in data_loader:
            #load IO dan pindah ke GPU
            rgbs = []
            segs = []
            deps = []
            # d_cld_xs = []
            # d_cld_zs = []
            for i in range(0, config.seq_len): #append data untuk input sequence
                rgbs.append(data['rgbs'][i].to(config.gpu_device, dtype=config.dtype))
                segs.append(data['rgb_segs'][i].to(config.gpu_device, dtype=config.dtype))
                deps.append(data['depths'][i].to(config.gpu_device, dtype=config.dtype))
                # check_gt_seg(config, segs[-1])
                # d_cld_xs.append(data['d_cld_xs'][i].to(config.gpu_device, dtype=config.dtype))
                # d_cld_zs.append(data['d_cld_zs'][i].to(config.gpu_device, dtype=config.dtype))
            # rp1 = torch.stack(data['rp1'], dim=1).to(config.gpu_device, dtype=config.dtype)
            # rp2 = torch.stack(data['rp2'], dim=1).to(config.gpu_device, dtype=config.dtype)
            # gt_velocity = torch.stack([data['velo']], dim=1).to(config.gpu_device, dtype=config.dtype)
            gt_waypoints = [torch.stack(data['waypoints'][j], dim=1).to(config.gpu_device, dtype=config.dtype) for j in range(0, config.pred_len)]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(config.gpu_device, dtype=config.dtype)
            gt_robot_ctrl = torch.stack(data['robot_ctrl'], dim=1).to(config.gpu_device, dtype=config.dtype)

            #forward pass
            

            #forward pass
            # velo = gt_velocity.cpu().detach().numpy()
            start_time = time.time() #waktu mulai
            pred_segs, pred_ctrl = model(rgbs, deps, data['cmd'])#, seg_fronts[-1])
            #ctrl_final, meta_pred = model.mlp_pid_control(pred_wp, pred_ctrl, velo)
            elapsed_time = time.time() - start_time #hitung elapsedtime

            #compute metric
            """
            metric_seg = 0
            for i in range(0, config.seq_len):
                metric_seg = metric_seg + IOU(pred_segs[i], segs[i])
            metric_seg = metric_seg / config.seq_len #dirata-rata
            """
            metric_seg = IOU(pred_segs[-1], segs[-1]) #ambil yang terakhir saja
            metric_ctrl = F.l1_loss(pred_ctrl, gt_robot_ctrl)
            total_metric = (1-metric_seg.item()) + metric_ctrl.item()

            #hitung rata-rata (avg) metric, dan metric untuk batch-batch yang telah diproses
            score['total_metric'].update(total_metric)#.item())
            score['ss_metric'].update(metric_seg.item()) 
            score['ctrl_metric'].update(metric_ctrl.item())

            #update visualisasi progress bar
            postfix = OrderedDict([('te_total_m', score['total_metric'].avg),
                                ('te_ss_m', score['ss_metric'].avg),
                                ('te_ctrl_m', score['ctrl_metric'].avg)])
            
            #simpan history test ke file csv, ambil dari hasil kalkulasi metric langsung, jangan dari averagemeter
            log['batch'].append(batch_ke)
            log['test_metric'].append(total_metric)#.item())
            log['test_ss_metric'].append(metric_seg.item())
            log['test_ctrl_metric'].append(metric_ctrl.item())
            log['elapsed_time'].append(elapsed_time)
            #paste ke csv file
            save_dir_log = save_dir+'outputs'
            os.makedirs(save_dir_log, exist_ok=True)
            pd.DataFrame(log).to_csv(save_dir_log+'/test_log.csv', index=False)

            #save outputnya
            """
            meta_output = get_wprp_frame(config, pred_wp[0].cpu().detach().numpy(),
                                         rp1[0].cpu().detach().numpy(), rp2[0].cpu().detach().numpy(),
                                         velo,
                                         pred_ctrl[0].cpu().detach().numpy(), data['bearing_robot'].item(),
                                         data['filename'][-1][-8:-4],
                                         [data['lat_robot'].item(), data['lon_robot'].item()])
            save_out(config, pred_segs[-1], pred_sdcs[-1], save_dir, data['route'][-1], data['filename'][-1], meta_output) #ambil pred_segs dan sdcs terakhir saja untuk diproses
            """
            
            #visualisasi seg dan SDC
            pred_seg = pred_segs[-1].cpu().detach().numpy() 
            imgx = colorize_seg(pred_seg, config.cityscapes_palette) #, add_bg=True
            save_dir_seg = save_dir+'outputs/'+data['route'][-1]+'/pred_seg/'
            os.makedirs(save_dir_seg, exist_ok=True)
            #print(save_dir_seg+data['filename'][-1]+"png")
            cv2.imwrite(save_dir_seg+data['filename'][-1]+"png", imgx) #cetak predicted segmentation

            save_dir_meta = save_dir+'outputs/'+data['route'][-1]+'/meta/'
            os.makedirs(save_dir_meta, exist_ok=True)
            """
            with open(save_dir_meta+data['filename'][-1]+"yml", 'w') as dict_file:
                yaml.dump(meta_pred, dict_file)
            """
            batch_ke += 1  
            prog_bar.set_postfix(postfix)
            prog_bar.update(1)
        prog_bar.close()
        
        #ketika semua sudah selesai, hitung rata2 performa pada log
        log['batch'].append("avg")
        log['test_metric'].append(np.mean(log['test_metric']))
        log['test_ss_metric'].append(np.mean(log['test_ss_metric']))
        log['test_ctrl_metric'].append(np.mean(log['test_ctrl_metric']))
        log['elapsed_time'].append(np.mean(log['elapsed_time']))
        
        #ketika semua sudah selesai, hitung VARIANCE performa pada log
        log['batch'].append("stddev")
        log['test_metric'].append(np.std(log['test_metric'][:-1]))
        log['test_ss_metric'].append(np.std(log['test_ss_metric'][:-1]))
        log['test_ctrl_metric'].append(np.std(log['test_ctrl_metric'][:-1]))
        log['elapsed_time'].append(np.std(log['elapsed_time'][:-1]))

        #paste ke csv file
        pd.DataFrame(log).to_csv(save_dir_log+'/test_log.csv', index=False)


    #return value
    return log



# Load config
config = GlobalConfig()

#SET GPU YANG AKTIF
torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id#visible_gpu #"0" "1" "0,1"

#IMPORT MODEL dan load bobot
print("IMPORT ARSITEKTUR DL DAN COMPILE")
model = huang(config).to(config.gpu_device, dtype=config.dtype)
model.load_state_dict(torch.load(os.path.join(config.logdir, 'best_model.pth')))

#BUAT DATA BATCH
test_set = Go2_Data(data_root=config.test_dir, config=config)
dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) #BS selalu 1

#test
test_log = test(dataloader_test, model, config)


#kosongkan cuda chace
torch.cuda.empty_cache()


import numpy as np
import torch
import os
from PIL import ImageFont

class GlobalConfig:
    bev_h = 128
    bev_w = 256
    front_h = 64
    front_w = 512
    cam_h = bev_h
    cam_w = bev_w
    #coverage_area = 16 #untuk top view SC, 24m kedepan, kiri, dan kanan
    cam_cover_area_lr = 8 #kiri - kanan
    cam_cover_area_rf = [0, 16] #posisi camera -> area interest max

    # w = 256
    hz = 1 #1 detik ada berapa sample yang direcord
    #bias_basic = 17
    #bearing_bias = [bias_basic, bias_basic+7, bias_basic, -bias_basic+5, -bias_basic+7, bias_basic] #dalam derajat #bias untuk 0 ke 50, 50 ke 120, 120 ke 180, -180 ke -120, -120 ke -60, -60 ke 0
    rp1_close = 1 #jarak minimum untuk ganti rp1 (dalam meter)   

    #for training
    gpu_id = '0'
    logdir = 'log/rb15'
    init_stop_counter = 40
    batch_size = 5
    lr = 1e-4 # learning rate #pakai AdamW
    lr_patience = 5 # nunggu berapa epoch sebelum lr diturunkan
    weight_decay = 1e-3
    #parameter untuk MGN
    MGN = True
    lw_alpha = 1.5
    #URUTAN BOTTLENECK: rgb enc, neck net
    bottleneck = [209, 500]
    loss_weights = [1, 1, 1, 1] #ss, de, wp, ctrl
    # n_fmap = [48, 96, 192, 384]
    n_fmap_b0 = [[32,16], [24], [40], [80,112], [192,320,1280]]
    n_fmap_b1 = [[32,16], [24], [40], [80,112], [192,320,1280]] #sama dengan b0
    n_fmap_b2 = [[32,16], [24], [48], [88,120], [208,352,1408]]
    n_fmap_b3 = [[40,24], [32], [48], [96,136], [232,384,1536]] #lihat underdevelopment/efficientnet.py
    n_fmap_b4 = [[48,24], [32], [56], [112,160], [272,448,1792]]

	# Data
    seq_len = 3 # jumlah input seq
    seq_gap = int(hz*3) #berapa frame-hz?
    pred_len = 5 # future waypoints predicted
    n_wp = pred_len #waypoints
    wp_gap = int(hz*5) #berapa frame-hz?
    gap_bearing = wp_gap #buat estimasi bearing berapa frame?
    logdir = logdir+"_seq"+str(seq_len)+"_pred"+str(pred_len) #update direktori name
    # root_dir = '/media/aisl/data/oskar/ros-whill-robot2/main/dataset/dataset'
    # root_dir = '/home/aisl/OSKAR/WHILL/ros-whill-robot2/main/dataset/dataset'
    root_dir = os.path.dirname(os.getcwd())+'/dataset/dataset'
    train_dir = root_dir+'/train_routes'
    val_dir = root_dir+'/val_routes'
    test_dir = root_dir+'/test_routes'


    #SETINGAN BUAT MODUL TRANSFUSER
    # n_views = 1 #DIDESAIN UNTUK 1 CAMERA VIEW AJA DULU
    vert_anchors = 4 #8 DIBUAT 4 (SEPARUH DARI HORIZONTAL ANCHOR) KARENA UKURAN INPUTNYA PERSEGI PANJANG HXW = 128X256
    horz_anchors = 8
    # anchors = vert_anchors * horz_anchors #GA DIPAKAI JUGA DI model_tf.py
    # n_embd = 512 #MENYESUAIKAN OUTPUT CHANNEL DARI EFFICIENT NET DISETIAP BLOCKNYA
    block_exp = 4
    n_layer = 8
    n_head = 4
    # n_scale = 4 #GA DIPAKAI JUGA DI model_tf.py
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1


    #buat preprocessing data
    polarseg_weight_path = os.path.join(os.getcwd(), "polarseg/SemKITTI_PolarSeg.pt")
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id#visible_gpu #"0" "1" "0,1"
    gpu_device = torch.device("cuda:0")
    dtype = torch.float32
    #cover_area_lr = 16 #kiri - kanan
    #cover_area_up = [-1, 7] #bawah -> atas
    #cover_area_f = [1.25, 17.25] #posisi sensor -> area interest max
    lid_cover_area_lr = 16 #kiri - kanan
    lid_cover_area_bt = [-1, 7] #bawah -> atas
    lid_cover_area_rf = [-16, 16] #posisi belakang lidar -> depan lidar
    SEG_CLASSES = { #lihat di file semantic-kitti.yaml
        'colors'        :[[0, 0, 0], [245, 150, 100], [245, 230, 100], [150, 60, 30], [180, 30, 80],
                        [255, 0, 0], [30, 30, 255], [200, 40, 255], [90, 30, 150],
                        [255, 0, 255], [255, 150, 255], [75, 0, 75], [75, 0, 175],
                        [0, 200, 255], [50, 120, 255], [0, 175, 0], [0, 60, 135],
                        [80, 240, 150], [150, 240, 255], [0, 0, 255]],  
        'classes'       : ['unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
                            'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 
                            'road', 'parking', 'sidewalk', 'other-ground', 
                            'building', 'fence', 'vegetation', 'trunk',
                            'terrain', 'pole', 'traffic-sign']
    }
    n_class_kitti = len(SEG_CLASSES['colors'])
    
    #lidar setting, cek HDL-32E dan VLP32C LiDAR sensor datasheet
    lidar_sensor = "mid360" #vlp32c hdl32e mid360
    if lidar_sensor == "hdl32e":
        v_fov = [-30.67, 10.67] # HDL32 pakai [-30.67, 10.67], VLP32 pakai [-25, 15]
        dep_max = 70#/1.25 #dalam meter, baca datasheet np.sqrt(cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (cover_area_up[1]-((cover_area_up[1]-cover_area_up[0])/2))**2)
        v_res_div = 60
    else: #"mid360"
        v_fov = [-25, 15] # HDL32 pakai [-30.67, 10.67], VLP32 pakai [-25, 15]
        dep_max = 70#/1.25 #dalam meter, baca datasheet np.sqrt(cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (cover_area_up[1]-((cover_area_up[1]-cover_area_up[0])/2))**2)
        v_res_div = 60
    max_intensity = 100.0
    # v_fov_down = -1*np.radians(2)
    # v_fov_up = np.radians(24.9)
    # n_laser = 32
    # lidar_rps = 10 #rotasi per detik --> 600 rpm / 60 detik
    h_fov = 360
    # v_fov = [-25, 15] # HDL32 pakai [-30.67, 10.67], VLP32 pakai [-25, 15]
    v_fov_total = -v_fov[0] + v_fov[1]

    v_res = v_fov_total/v_res_div         #n_laser #front_h  # 1.33 #vertical resolution
    h_res = h_fov/(front_w*2)              #0.35 #horizontal resolution
    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)
    # y_fudge = 5

    #config polarseg
    ignore_label = 0
    grid_size = np.asarray([480,360,32])
    max_volume_space = np.asarray([50,np.pi,1.5])
    min_volume_space = np.asarray([3,-np.pi,-3])
    intervals = (max_volume_space - min_volume_space) / (grid_size-1)
    #untuk operasi langsung tensor
    grid_size_ten = torch.from_numpy(np.asarray([480,360,32])).to(gpu_device, dtype=dtype)
    max_volume_space_ten = torch.from_numpy(np.asarray([50,np.pi,1.5])).to(gpu_device, dtype=dtype)
    min_volume_space_ten = torch.from_numpy(np.asarray([3,-np.pi,-3])).to(gpu_device, dtype=dtype)
    intervals_ten = (max_volume_space_ten - min_volume_space_ten) / (grid_size_ten-1)
    #untuk front_dep dan bev_dep
    #100 untuk HDL32E, 200 untuk VLP32C
    # dep_max = 200#/1.25 #dalam meter, baca datasheet np.sqrt(cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (cover_area_up[1]-((cover_area_up[1]-cover_area_up[0])/2))**2)
    dep_min = 0#lid_cover_area_rf[0]

    #settingan segformer
    segformer_weight_path = os.path.join(os.getcwd(), "segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth")
    segformer_config_path = os.path.join(os.getcwd(), "segformer/configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py")
    #BACA https://mmsegmentation.readthedocs.io/en/latest/_modules/mmseg/core/evaluation/class_names.html#get_palette
    #HANYA ADA 19 CLASS?? #tambahan 0,0,0 hitam untuk area kosong pada SDC nantinya
    cityscapes_palette = [[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],  
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], 
            [0, 80, 100], [0, 0, 230], [119, 11, 32]] #,  
    n_class_cityscape = len(cityscapes_palette)
    cityscapes_classes = ['None', 'road', 'sidewalk', 'building', 'wall',
                            'fence', 'pole', 'traffic light', 'traffic sign', 
                            'vegetation', 'terrain', 'sky', 'person', 
                            'rider', 'car', 'truck', 'bus',
                            'train', 'motorcycle', 'bicycle']


    #other, buat join_img dll
    fontsize = 14
    font_mul = 1
    fontx = ImageFont.truetype(font="arial.ttf", size=font_mul*fontsize) #arialbold arial
    #jika error font tidak ketemu, donlod dulu di https://www.freefontspro.com/14454/arial.ttf lalu copas foldernya ke /usr/share/fonts/truetype/
    # fontx = ImageFont.load_default()
    text_gap = (fontsize+2)*font_mul
    metadata_gap = 95*font_mul
    
    fps = 20
    rgb_res_ori = [720, 1280] #HxW
    scale_w = rgb_res_ori[1]/front_w
    scaled_H_rgb = int(rgb_res_ori[0]/scale_w)

    
    # Controller
    #control weights untuk PID dan MLP dari tuningan MGN
    #baca dulu trainval_log.csv setelah training selesai, dan normalize bobotnya 0-1
    #LWS: lw_wp lw_ctrl saat convergence
    lws = [1, 1]
    cw_pid = lws[0]/(lws[0]+lws[1]) # untuk ypos, orient
    cw_mlp = 1-cw_pid # untuk ypos, xpos, orient

    turn_KP = 0.5
    turn_KI = 0.25
    turn_KD = 0.15
    turn_n = 15 # buffer size

    speed_KP = 1.5
    speed_KI = 0.25
    speed_KD = 0.5
    speed_n = 15 # buffer size

    n_cmd = 3 #jumlah command yang ada: 0 lurus, 1 kiri, 2 kanan
    max_throttle = 1.0 # upper limit on throttle signal value in dataset
    wheel_radius = 0.15#radius roda robot dalam meter
    # brake_speed = 0.4 # desired speed below which brake is triggered
    # brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    # clip_delta = 0.25 # maximum change in speed input to logitudinal controller
    min_act_thrt = 0.1 #minimum nilai suatu throttle dianggap aktif diinjak
    err_angle_mul = 0.075
    des_speed_mul = 1.75
    """"""


    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

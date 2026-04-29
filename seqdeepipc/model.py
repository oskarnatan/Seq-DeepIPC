from collections import deque
import sys
import numpy as np
from torch import torch, cat, nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import math



#FUNGSI INISIALISASI WEIGHTS MODEL
#baca https://pytorch.org/docs/stable/nn.init.html
#kaiming he
def kaiming_init(m):
    # print(m)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # m.bias.data.fill_(0.01)


class ConvBNRelu(nn.Module):
    def __init__(self, channelx, stridex=1, kernelx=3, paddingx=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(channelx[0], channelx[1], kernel_size=kernelx, stride=stridex, padding=paddingx, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(channelx[1])
        self.relu = nn.ReLU()
        #weights initialization
        # kaiming_w_init(self.conv)
    
    def forward(self, x):
        x = self.conv(x) 
        x = self.bn(x) 
        y = self.relu(x)
        return y


class ConvBlock(nn.Module):
    def __init__(self, channel, final=False, task='seg'): #up, 
        super(ConvBlock, self).__init__()
        #conv block
        if final:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[0]], stridex=1)
            if task == 'seg':
                self.conv_block1 = nn.Sequential(
                nn.Conv2d(channel[0], channel[1], kernel_size=1),
                nn.Sigmoid()
                )
            else:
                self.conv_block1 = nn.Sequential(
                nn.Conv2d(channel[0], channel[1], kernel_size=1),
                nn.ReLU()
                )  
        else:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[1]], stridex=1)
            self.conv_block1 = ConvBNRelu(channelx=[channel[1], channel[1]], stridex=1)
        #init
        self.conv_block0.apply(kaiming_init)
        self.conv_block1.apply(kaiming_init)
 
    def forward(self, x):
        #convolutional block
        y = self.conv_block0(x)
        y = self.conv_block1(y)
        return y



class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0
    
    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        out_control = self._K_P * error + self._K_I * integral + self._K_D * derivative
        return out_control




class rb15(nn.Module): #
    #default input channel adalah 3 untuk RGB, 2 untuk DVS, 1 untuk LiDAR
    def __init__(self, config):#n_fmap, n_class=[23,10], n_wp=5, in_channel_dim=[3,2], spatial_dim=[240, 320], gpu_device=None): 
        super(rb15, self).__init__()
        self.config = config
        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        #------------------------------------------------------------------------------------------------
        #RGB, jika inputnya sequence, maka jumlah input channel juga harus menyesuaikan
        self.rgb_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.efficientnet_b0(pretrained=True) #efficientnet_b4
        self.RGB_encoder.classifier = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        self.RGB_encoder.avgpool = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan 
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 

        #SS
        self.conv3_ss_f = ConvBlock(channel=[config.n_fmap_b0[4][-1]+config.n_fmap_b0[3][-1], config.n_fmap_b0[3][-1]])#, up=True)
        self.conv2_ss_f = ConvBlock(channel=[config.n_fmap_b0[3][-1]+config.n_fmap_b0[2][-1], config.n_fmap_b0[2][-1]])#, up=True)
        self.conv1_ss_f = ConvBlock(channel=[config.n_fmap_b0[2][-1]+config.n_fmap_b0[1][-1], config.n_fmap_b0[1][-1]])#, up=True)
        self.conv0_ss_f = ConvBlock(channel=[config.n_fmap_b0[1][-1]+config.n_fmap_b0[0][-1], config.n_fmap_b0[0][0]])#, up=True)
        self.final_ss_f = ConvBlock(channel=[config.n_fmap_b0[0][0], config.n_class_cityscape], final=True, task='seg')#, up=False)
        #DE
        self.conv3_de_f = ConvBlock(channel=[config.n_fmap_b0[4][-1]+config.n_fmap_b0[3][-1], config.n_fmap_b0[3][-1]])#, up=True)
        self.conv2_de_f = ConvBlock(channel=[config.n_fmap_b0[3][-1]+config.n_fmap_b0[2][-1], config.n_fmap_b0[2][-1]])#, up=True)
        self.conv1_de_f = ConvBlock(channel=[config.n_fmap_b0[2][-1]+config.n_fmap_b0[1][-1], config.n_fmap_b0[1][-1]])#, up=True)
        self.conv0_de_f = ConvBlock(channel=[config.n_fmap_b0[1][-1]+config.n_fmap_b0[0][-1], config.n_fmap_b0[0][0]])#, up=True)
        self.final_de_f = ConvBlock(channel=[config.n_fmap_b0[0][0], 1], final=True, task='dep')#, up=False)
        #------------------------------------------------------------------------------------------------

        #untuk semantic cloud generator
        #self.cover_area = config.coverage_area
        self.n_class = config.n_class_cityscape
        self.h, self.w = config.bev_h, config.bev_w
        #SC
        self.SC_encoder = models.efficientnet_b0(pretrained=False) #efficientnet_b0
        self.SC_encoder.features[0][0] = nn.Conv2d(config.n_class_cityscape, config.n_fmap_b0[0][0], kernel_size=3, stride=2, padding=1, bias=False) #ganti input channel conv pertamanya, buat SC cloud
        self.SC_encoder.classifier = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        self.SC_encoder.avgpool = nn.Sequential()
        self.SC_encoder.apply(kaiming_init)
        #------------------------------------------------------------------------------------------------
        #feature fusion
        self.necks_net = nn.Sequential( #inputnya dari 2 bottleneck
            nn.Conv2d(config.n_fmap_b0[4][-1]+config.n_fmap_b0[4][-1], config.n_fmap_b0[4][1], kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap_b0[4][1], config.n_fmap_b0[4][0])
        )
        #------------------------------------------------------------------------------------------------
        #wp predictor, input size 7 karena concat dari wp xy, rp1 xy, rp2 xy, dan velocity
        self.gru = nn.GRUCell(input_size=7, hidden_size=config.n_fmap_b0[4][0])
        self.pred_dwp = nn.Linear(config.n_fmap_b0[4][0], 2)

        #------------------------------------------------------------------------------------------------
        #controller
        #MLP Controller

        ##MLP Controller ada 3, 0 = lurus, 1 = belok kiri, 2 = belok kanan
        """
        self.mlp_controller = nn.Sequential( 
            nn.Linear(config.n_fmap_b0[2], config.n_fmap_b0[1]),
            nn.Linear(config.n_fmap_b0[1], 4), #x pos, y pos, x orient, y orient
            nn.ReLU()
        )
        """
        self.mlp_controller = nn.ModuleList([nn.Sequential( 
            #nn.Linear(config.n_fmap_b0[4][0], config.n_fmap_b0[4][0]),
            nn.Linear(config.n_fmap_b0[4][0], 3), #x pos, y pos, x orient, y orient
            nn.ReLU()
        ) for _ in range(config.n_cmd)]) #.to(self.gpu_device, dtype=torch.float)
        
        #PID Controller
        self.orient_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.y_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        self.x_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)



    def forward(self, rgbs, pt_cloud_xs, pt_cloud_zs, rp1, rp2, velo_in, cmd):#, gt_ss): , 
        #------------------------------------------------------------------------------------------------
        #bagian downsampling
        RGB_features_sum = 0
        SC_features_sum = 0
        segs_f = []
        des_f = []
        sdcs = []
        for i in range(self.config.seq_len): #loop semua input dalam buffer
            in_rgb = self.rgb_normalizer(rgbs[i]) #
            RGB_features0 = self.RGB_encoder.features[0](in_rgb)
            RGB_features1 = self.RGB_encoder.features[1](RGB_features0)
            RGB_features2 = self.RGB_encoder.features[2](RGB_features1)
            RGB_features3 = self.RGB_encoder.features[3](RGB_features2)
            RGB_features4 = self.RGB_encoder.features[4](RGB_features3)
            RGB_features5 = self.RGB_encoder.features[5](RGB_features4)
            RGB_features6 = self.RGB_encoder.features[6](RGB_features5)
            RGB_features7 = self.RGB_encoder.features[7](RGB_features6)
            RGB_features8 = self.RGB_encoder.features[8](RGB_features7)
            RGB_features_sum += RGB_features8
            #bagian upsampling
            rgbneck = cat([self.up(RGB_features8), RGB_features5], dim=1)
            #ss
            ss_f_3 = self.conv3_ss_f(rgbneck)
            ss_f_2 = self.conv2_ss_f(cat([self.up(ss_f_3), RGB_features3], dim=1))
            ss_f_1 = self.conv1_ss_f(cat([self.up(ss_f_2), RGB_features2], dim=1))
            ss_f_0 = self.conv0_ss_f(cat([self.up(ss_f_1), RGB_features1], dim=1))
            ss_f = self.final_ss_f(self.up(ss_f_0))
            segs_f.append(ss_f)
            #de
            de_f_3 = self.conv3_de_f(rgbneck)
            de_f_2 = self.conv2_de_f(cat([self.up(de_f_3), RGB_features3], dim=1))
            de_f_1 = self.conv1_de_f(cat([self.up(de_f_2), RGB_features2], dim=1))
            de_f_0 = self.conv0_de_f(cat([self.up(de_f_1), RGB_features1], dim=1))
            de_f = self.final_de_f(self.up(de_f_0))
            des_f.append(de_f)
            #------------------------------------------------------------------------------------------------
            #buat semantic cloud
            top_view_sc = self.gen_top_view_sc_ptcloud(pt_cloud_xs[i], pt_cloud_zs[i], ss_f)
            sdcs.append(top_view_sc)
            #bagian downsampling
            SC_features0 = self.SC_encoder.features[0](top_view_sc)
            SC_features1 = self.SC_encoder.features[1](SC_features0)
            SC_features2 = self.SC_encoder.features[2](SC_features1)
            SC_features3 = self.SC_encoder.features[3](SC_features2)
            SC_features4 = self.SC_encoder.features[4](SC_features3)
            SC_features5 = self.SC_encoder.features[5](SC_features4)
            SC_features6 = self.SC_encoder.features[6](SC_features5)
            SC_features7 = self.SC_encoder.features[7](SC_features6)
            SC_features8 = self.SC_encoder.features[8](SC_features7)
            SC_features_sum += SC_features8

        #------------------------------------------------------------------------------------------------
        #waypoint prediction
        #get hidden state dari gabungan kedua bottleneck
        hx = self.necks_net(cat([RGB_features_sum, SC_features_sum], dim=1))
        # initial input car location ke GRU, selalu buat batch size x 2 (0,0) (xy)
        xy = torch.zeros(size=(hx.shape[0], 2)).to(self.config.gpu_device, dtype=hx.dtype)
        #predict delta wp
        out_wp = list()
        for _ in range(self.config.pred_len):
            ins = torch.cat([xy, rp1, rp2, velo_in], dim=1)
            hx = self.gru(ins, hx)
            d_xy = self.pred_dwp(hx)
            xy = xy + d_xy
            out_wp.append(xy)
            # if nwp == 1: #ambil hidden state ketika sampai pada wp ke 2, karena 3, 4, dan 5 sebenarnya tidak dipakai
            #     hx_mlp = torch.clone(hx)
        pred_wp = torch.stack(out_wp, dim=1)
        #------------------------------------------------------------------------------------------------
        #control decoder
        """
        control_pred = self.mlp_controller(hx) #cat([hid_states, hid_state_nxr, hid_state_vel], dim=1)

        """
        #control decoder #cmd ada 3, 0 = lurus, 1 = belok kiri, 2 = belok kanan
        #sementara ini terpaksa loop sepanjang batch dulu, ga tau caranya supaya langsung
        # print(cmd)
        # print(len(cmd))
        # print(cmd.shape)
        control_pred = self.mlp_controller[cmd[0].item()](hx[0:1,:])
        for i in range(1, len(cmd)): 
            # print("-----------")
            # print(cmd[i].item())
            # print(hx[i:i+1,:].shape)
            control_pred = cat([control_pred, self.mlp_controller[cmd[i].item()](hx[i:i+1,:])], dim=0) #concat di axis batch
            # print(control_pred.shape)
        #denormalisasi
        # print(control_pred.shape)
        

        # position control (left xy joystic)
        control_pred[:,0] = control_pred[:,0] * 2 - 1.0 # convert from [0,1] to [-1,1]
        control_pred[:,1] = control_pred[:,1]
        # orientation control (right x joystic)
        control_pred[:,2] = control_pred[:,2] * 2 - 1.0 # convert from [0,1] to [-1,1]
        #control_pred[:,3] = control_pred[:,3] * 2 - 1. # convert from [0,1] to [-1,1]

        return segs_f, des_f, pred_wp, control_pred, sdcs


    def gen_top_view_sc_ptcloud(self, pt_cloud_x, pt_cloud_z, semseg):
        #proses awal
        _, label_img = torch.max(semseg, dim=1) #pada axis C
        cloud_data_n = torch.ravel(torch.tensor([[n for _ in range(self.h*self.w)] for n in range(semseg.shape[0])])).to(self.config.gpu_device, dtype=semseg.dtype)

        #normalize ke frame 
        cloud_data_x = torch.round((pt_cloud_x + self.config.cam_cover_area_lr) * (self.w-1) / (2*self.config.cam_cover_area_lr)).ravel()
        cloud_data_z = torch.round((pt_cloud_z * (1-self.h) / (self.config.cam_cover_area_rf[1]-self.config.cam_cover_area_rf[0])) + (self.h-1)).ravel()

        #cari index interest
        bool_xz = torch.logical_and(torch.logical_and(cloud_data_x <= self.w-1, cloud_data_x >= 0), torch.logical_and(cloud_data_z <= self.h-1, cloud_data_z >= 0))
        idx_xz = bool_xz.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya

        #stack n x z cls dan plot
        # print(cloud_data_n.shape)
        # print(label_img.ravel().shape)
        # print(cloud_data_z.shape)
        # print(cloud_data_x.shape)
        coorx = torch.stack([cloud_data_n, label_img.ravel(), cloud_data_z, cloud_data_x])
        coor_clsn = torch.unique(coorx[:, idx_xz], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
        top_view_sc = torch.zeros_like(semseg) #ini lebih cepat karena secara otomatis size, tipe data, dan device sama dengan yang dimiliki inputnya (semseg)
        #print(top_view_sc.shape)
        top_view_sc[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #format axis dari NCHW

        return top_view_sc





    def mlp_pid_control(self, pwaypoints, pctrl, linear_velo):
        assert(pwaypoints.size(0)==1)
        waypoints = pwaypoints[0].data.cpu().numpy()
        
        #vehicular controls dari PID
        aim_point = (waypoints[1] + waypoints[0]) / 2.0 #tengah2nya wp0 dan wp1
        #90 deg ke kanan adalah 0 radian, 90 deg ke kiri adalah 1*pi radian
        angle_rad = np.clip(np.arctan2(aim_point[1], aim_point[0]), 0, np.pi) #arctan y/x
        angle_deg = np.degrees(angle_rad)
        #ke kiri adalah 0 -> +1 == 90 -> 180, ke kanan adalah 0 -> -1 == 90 -> 0
        error_angle = (angle_deg - 90.0) * self.config.err_angle_mul
        pid_orientation = self.orient_controller.step(error_angle)
        pid_orientation = np.clip(pid_orientation, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[1] - waypoints[0]) * self.config.des_speed_mul
        #linear_velo = np.mean(angular_velo) * self.config.wheel_radius
        #delta = np.clip(desired_speed - linear_velo, 0.0, self.config.clip_delta)
        pid_ypos = self.y_controller.step(desired_speed - linear_velo)
        pid_ypos = np.clip(pid_ypos, 0.0, self.config.max_throttle)

        #proses vehicular controls dari MLP
        mlp_orientation = np.clip(pctrl.cpu().data.numpy()[0][2], -1.0, 1.0)
        mlp_ypos = np.clip(pctrl.cpu().data.numpy()[0][1], 0.0, self.config.max_throttle)
        mlp_xpos = np.clip(pctrl.cpu().data.numpy()[0][0], -1.0, 1.0)

        #opsi 1: jika salah satu controller aktif, maka vehicle jalan. vehicle berhenti jika kedua controller non aktif
        act_pid_ypos = pid_ypos >= self.config.min_act_thrt
        act_mlp_ypos = mlp_ypos >= self.config.min_act_thrt
        if act_pid_ypos and act_mlp_ypos:
            act_pid_orientation = np.abs(pid_orientation) >= self.config.min_act_thrt
            act_mlp_orientation = np.abs(mlp_orientation) >= self.config.min_act_thrt
            if act_pid_orientation and not act_mlp_orientation:
                orien_ctrl = pid_orientation
            elif act_mlp_orientation and not act_pid_orientation:
                orien_ctrl = mlp_orientation
            else: #keduanya sama2 kurang dari threshold atau sama2 lebih dari threshold
                orien_ctrl = 0.5*pid_orientation + 0.5*mlp_orientation
            ypos_ctrl = 0.5*pid_ypos + 0.5*mlp_ypos
        elif act_pid_ypos and not act_mlp_ypos:
            orien_ctrl = pid_orientation
            ypos_ctrl = pid_ypos
        elif act_mlp_ypos and not act_pid_ypos:
            orien_ctrl = mlp_orientation
            ypos_ctrl = mlp_ypos
        else: # (pid_ypos < self.config.min_act_thrt) and (mlp_ypos < self.config.min_act_thrt):
            orien_ctrl = 0.0 #dinetralkan
            ypos_ctrl = 0.0
        orien_ctrl = float(orien_ctrl)
        ypos_ctrl = float(ypos_ctrl)
        xpos_ctrl = float(mlp_xpos)

        # print(waypoints[2])
        
        metadata = {
            'orien_ctrl': orien_ctrl,
            'ypos_ctrl': ypos_ctrl,
            'xpos_ctrl': xpos_ctrl,
            'linear_velo' : float(linear_velo),
            'cw_pid': float(0.5), #self.config.cw_pid
            'pid_orientation': float(pid_orientation),
            'pid_ypos': float(pid_ypos),
            'cw_mlp': float(0.5), #self.config.cw_mlp
            'mlp_orientation': float(mlp_orientation),
            'mlp_ypos': float(mlp_ypos),
            'mlp_xpos': float(mlp_xpos),
            'wp_5': [float(waypoints[4][0].astype(np.float64)), float(waypoints[4][1].astype(np.float64))], #tambahan
            'wp_4': [float(waypoints[3][0].astype(np.float64)), float(waypoints[3][1].astype(np.float64))], #tambahan
            'wp_3': [float(waypoints[2][0].astype(np.float64)), float(waypoints[2][1].astype(np.float64))], #tambahan
            'wp_2': [float(waypoints[1][0].astype(np.float64)), float(waypoints[1][1].astype(np.float64))],
            'wp_1': [float(waypoints[0][0].astype(np.float64)), float(waypoints[0][1].astype(np.float64))],
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle_deg.astype(np.float64)),
            'aim': [float(aim_point[0].astype(np.float64)), float(aim_point[1].astype(np.float64))],
            # 'delta': float(delta.astype(np.float64)),
            'robot_pos_global': None, #akan direplace nanti
            'robot_bearing': None, #akan direplace nanti
            'rp1_pos_global': None, #akan direplace nanti
            'rp2_pos_global': None, #akan direplace nanti
            'rp1_pos_local': None, #akan direplace nanti
            'rp2_pos_local': None, #akan direplace nanti
            'cmd': None, #akan direplace nanti
            'fps': None, #akan direplace nanti
            'model_fps': None, #akan direplace nanti
            'intervention': False, #akan direplace nanti
        }
        return orien_ctrl, ypos_ctrl, xpos_ctrl, metadata
""""""
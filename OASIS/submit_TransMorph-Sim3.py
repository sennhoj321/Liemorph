import glob, sys
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from surface_distance import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from scipy.ndimage.interpolation import map_coordinates, zoom
from models.TransMorph_flow import TransMorphFlow, TransMorphFlowSE3, TransMorphFlowSim3, Bilinear
import ml_collections
from lietorch import SO3, SE3, Sim3
from utils import OlddetJac
import torch.nn as nn

def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet

def get_3DTransMorph_config():
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 7, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

'''
GPU configuration
'''
GPU_iden = 1
GPU_num = torch.cuda.device_count()
print('Number of GPU: ' + str(GPU_num))
for GPU_idx in range(GPU_num):
    GPU_name = torch.cuda.get_device_name(GPU_idx)
    print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
torch.cuda.set_device(GPU_iden)
    
def main():
    test_dir = '/data/OASIS_L2R_2021_task03/Test/'
    save_dir = '/results/Quantitative_Results/'
    model_idx = -1
    weights = [1, 1, 2]
    model_folder =  'TransMorph_ncc_{}_dsc{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    model_dir = '/results/experiments/' + model_folder
    '''
    Initialize model
    '''
    config =  get_3DTransMorph_config()
    model = TransMorphFlowSE3(config)
    model.cuda()
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx],weights_only=False)['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()    
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    file_names = glob.glob(test_dir + '*.pkl')
    
    val_set = datasets.OASISBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=val_composed)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    
    '''
    Initialize spatial transformation function
    '''
    print(config.img_size)
    reg_model = utils.register_modelSE3(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_modelSE3(config.img_size, 'bilinear')
    reg_model_bilin.cuda()
    '''
    Validation
    '''
    eval_dsc = utils.AverageMeter()
    eval_hdd = utils.AverageMeter()
    eval_detjac = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in val_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            x_in = torch.cat((x, y), dim=1)
            #file_name = file_names[stdy_idx].split('\\')[-1].split('.')[0][2:]
            output = model(x_in)
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=36)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            output = model(x_in)
            loss = 0
            loss_vals = []
            x_segs = []
            for i in range(36):
                def_seg = reg_model_bilin([x_seg_oh[:, i:i + 1, ...].float(), output[2]])
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
            del x_segs, x_seg_oh
            eval_dsc.update(dsc.item(), x.size(0))
            print(eval_dsc.avg, eval_dsc.std)
            # hd95
            hd95 = 0
            count = 0
            #print(def_out.shape)
            # dice
            '''dice = 0 to test if it gives the same values
            count = 0
            for i in range(1, 36):
                if ((def_out==i).sum()==0) or ((y_seg==i).sum()==0):
                    continue
                dice += compute_dice_coefficient((def_out[0,0]==i).cpu().detach().numpy(), (y_seg[0,0]==i).cpu().detach().numpy())
                count += 1
            dice /= count
            print(dice)'''
            #detjac
            MetriK_NegJAc = utils.OlddetJac()
            metrik_negjac = MetriK_NegJAc.loss(output[3]).detach().cpu()
            log_jac_det = (np.log((metrik_negjac).clip(0.000000001, 1000000000))).std() # clipping taken from TransMorph
            eval_detjac.update(torch.mean((metrik_negjac<0).float()), x.size(0))
            print('detJac:', eval_detjac.avg, eval_detjac.std)
            print('detJac<0:',  torch.mean((metrik_negjac<0).float()))
            # hd95
            hd95 = 0
            count = 0
            for i in range(1, 36):
                if ((def_out==i).sum()==0) or ((y_seg==i).sum()==0):
                    continue
                hd95 += compute_robust_hausdorff(compute_surface_distances((def_out[0,0]==i).cpu().detach().numpy(), (y_seg[0,0]==i).cpu().detach().numpy(), np.ones(3)), 95.)
                count += 1
            hd95 /= count
            eval_hdd.update(hd95.item(), x.size(0))
            print(eval_hdd.avg, eval_hdd.std)
        #best_dsc = max(eval_dsc.avg, best_dsc)
        #np.savez(save_dir+'disp_{}.npz'.format(file_name), flow)
        stdy_idx += 1

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph_flow import TransMorphFlow, TransMorphFlowSE3, TransMorphFlowSim3, Bilinear
import ml_collections

import torch.nn as nn

def get_3DTransMorphLarge_config():
    '''
    A Large TransMorph Network
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 128
    config.depths = (2, 2, 12, 2)
    config.num_heads = (4, 4, 8, 16)
    config.window_size = (5, 6, 7)
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

def main():
    atlas_dir = '/data/IXI_data/atlas.pkl'
    test_dir = '/data/IXI_data/Test/'
    model_idx = -1
    #model_folder = 'TransMorphDiff_NCC_flowreg2/'
    model_folder= 'TransMorphDiff_NCC_SEflowreg2/'
    model_dir = '/results/experiments/' + model_folder
    if 'Val' in test_dir:
        csv_name = model_folder[:-1]+'_Val'
    else:
        csv_name = model_folder[:-1]
    dict = utils.process_label()
    if not os.path.exists('Results/Quantitative_Results/'):
        os.makedirs('Results/Quantitative_Results/')
    if os.path.exists('Results/Quantitative_Results/'+csv_name+'.csv'):
        os.remove('Results/Quantitative_Results/'+csv_name+'.csv')
    csv_writter(model_folder[:-1], '/results/Quantitative_Results/' + csv_name)
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line + ',' + 'non_jec', 'Results/Quantitative_Results/' + model_folder[:-1])
    config = get_3DTransMorph_config()
    model = TransMorphFlowSE3(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    '''
    Initialize spatial transformation function
    '''
    print(config.img_size)
    reg_model = utils.register_modelSE3(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_modelSE3(config.img_size, 'bilinear')
    reg_model_bilin.cuda()
    
    for param in reg_model.parameters():
        param.requires_grad = False
        param.volatile = True
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            x_in = torch.cat((x,y), dim=1)
            x_def, flow, disp_field = model(x_in)
            #def_out = reg_model([x_seg.cuda().float(), disp_field.cuda()])
            #x_segs = model.spatial_trans(x_seg.float(), flow.float())
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            x_segs = []
            for i in range(46):
                def_seg, new_locs = reg_model_bilin([x_seg_oh[:, i:i + 1, ...].float(),  disp_field.float()])
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            del x_segs, x_seg_oh
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            #jac_det = utils.jacobian_determinant_vxm(disp_field.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            MetriK_NegJAc = utils.OlddetJac()
            metrik_negjac = MetriK_NegJAc.loss(new_locs).detach().cpu()
            line = line  +','+str(metrik_negjac)
            csv_writter(line, '/results/Quantitative_Results/' + model_folder[:-1])
            eval_det.update(metrik_negjac, x.size(0))
            print('det < 0: {}'.format(metrik_negjac))
            dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
            dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

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

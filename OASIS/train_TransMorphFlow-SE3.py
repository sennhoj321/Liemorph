from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import matplotlib.pyplot as plt
import torch.nn as nn
from natsort import natsorted
from models.TransMorph_flow import TransMorphFlow, TransMorphFlowSE3, TransMorphFlowSim3, Bilinear
import ml_collections
from utils import register_modelSE3, register_modelSim3
from lietorch import SO3, SE3, Sim3

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
class GlobNCC:
    """
    Global normalized cross correlation loss.
    """

    def __init__(self, win=None, ndims=3):
        # compute filters
        # set window size
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        win = [9] * ndims if win is None else win
        self.win = win
        self.sum_filt = torch.ones([1, 1, *win]).to('cuda')

    def loss(self, y_true, y_pred, ndims=3):
        if len(y_true.size()) >= 3:
            H, W, Z = y_true.size()[-3:]
            Ii = y_true.reshape(1, -1, H, W, Z)
            Ji = y_pred.reshape(1, -1, H, W, Z)
        else:
            Ii = y_true
            Ji = y_pred

        # compute CC squares
        Ii, Ji = Ii - torch.mean(Ii, dim=(2, 3, 4), keepdim=True), Ji - torch.mean(Ji, dim=(2, 3, 4), keepdim=True)
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = torch.sum(Ii * Ji, dim=(2, 3, 4), keepdim=True)
        cc = IJ * IJ / (torch.sum(I2, dim=(2, 3, 4), keepdim=True) * torch.sum(J2, dim=(2, 3, 4), keepdim=True) + 1e-7)
        return 1 - torch.mean(cc)

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

class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)
    
def main():
    batch_size = 1
    train_dir = '/data/OASIS_L2R_2021_task03/All/'
    val_dir = '/data/OASIS_L2R_2021_task03/Test/'
    weights = [1, 1, 4] # loss weights
    save_dir = 'TransMorph_ncc_{}_dsc{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    if not os.path.exists('/results/experiments/'+save_dir):
        os.makedirs('/results/experiments/'+save_dir)
    if not os.path.exists('/results/logs/'+save_dir):
        os.makedirs('/results/logs/'+save_dir)
    sys.stdout = Logger('/results/logs/'+save_dir)
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 500 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    config =  get_3DTransMorph_config()
    model = TransMorphFlowSE3(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    print(config.img_size)
    reg_model = utils.register_modelSE3(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_modelSE3(config.img_size, 'bilinear')
    reg_model_bilin.cuda()
    


    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 61
        model_dir = '/results/experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-2])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),
                                         ])

    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.OASISBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    val_set = datasets.OASISBrainInferDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion_ncc = losses.NCC_vxm()
    criterion_dsc = losses.DiceLoss()
    criterion_reg = losses.Grad3d(penalty='l2')
    best_dsc = 0
    writer = SummaryWriter(log_dir='/results/tensorboard/NCC_SE3_flowreg4.0')
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            x_in = torch.cat((x,y), dim=1)
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=36)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            output = model(x_in)
            loss = 0
            loss_vals = []
            def_segs = []
            for i in range(36):
                def_seg,_,_ = model.spatial_transformer(x_seg_oh[:, i:i + 1, ...].float(), SE3.exp(output[2].reshape(-1,6)),device='cuda')
                def_segs.append(def_seg)
            def_seg = torch.cat(def_segs, dim=1)
            loss_ncc = criterion_ncc(output[0], y) * weights[0]
            loss_dsc = criterion_dsc(def_seg, y_seg.long()) * weights[1]
            loss_reg = criterion_reg(output[1], y) * weights[2]
            loss = loss_ncc + loss_dsc + loss_reg
            loss_all.update(loss.item(), y.numel())
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_seg_oh, x_in, def_segs, def_seg, loss

            y_seg_oh = nn.functional.one_hot(y_seg.long(), num_classes=36)
            y_seg_oh = torch.squeeze(y_seg_oh, 1)
            y_seg_oh = y_seg_oh.permute(0, 4, 1, 2, 3).contiguous()

            y_in = torch.cat((y, x), dim=1)
            output = model(y_in)
            def_segs = []
            for i in range(36):
                def_seg,_,_ = model.spatial_transformer(y_seg_oh[:, i:i + 1, ...].float(), SE3.exp(output[2].reshape(-1,6)),device='cuda')
                def_segs.append(def_seg)
            def_seg = torch.cat(def_segs, dim=1)
            loss_ncc = criterion_ncc(output[0], x) * weights[0]
            loss_dsc = criterion_dsc(def_seg, x_seg.long()) * weights[1]
            loss_reg = criterion_reg(output[1], x) * weights[2]
            loss = loss_ncc + loss_dsc + loss_reg
            loss_all.update(loss.item(), x.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del y_seg_oh, y_in, def_segs, def_seg
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, DSC: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                                loss.item(),
                                                                                                loss_ncc.item(),
                                                                                                loss_dsc.item(),
                                                                                                loss_reg.item()))
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, config.img_size)
                output = model(x_in)
                def_out = reg_model([x_seg.cuda().float(), output[2].cuda()])
                def_grid = reg_model_bilin([grid_img.float(), output[2].cuda()])
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='/results/experiments/'+save_dir, filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
        del def_out, def_grid, grid_img, output
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 3
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

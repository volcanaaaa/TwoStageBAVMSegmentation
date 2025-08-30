import os, argparse
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import sys
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# 添加模型定义代码
def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock2D, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock2D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock2D(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock2D(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=False):
        super(UpBlock2D, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock2D(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Encoder2D(nn.Module):
    def __init__(self, params):
        super(Encoder2D, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock2D(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock2D(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock2D(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock2D(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock2D(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class Decoder2D(nn.Module):
    def __init__(self, params):
        super(Decoder2D, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock2D(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock2D(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock2D(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock2D(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

class UNet2D(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet2D, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder2D(params)
        self.decoder = Decoder2D(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

class ChannelAttentionModule3D(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        attention = self.sigmoid(avgout + maxout)
        return attention, attention * x

class SpatialAttentionModule3D(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        attention = self.sigmoid(self.conv3d(out))
        return attention, attention * x

class CBAM3D(nn.Module):
    def __init__(self, channel):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttentionModule3D(channel)
        self.spatial_attention = SpatialAttentionModule3D()

    def forward(self, x):
        ca_att, ca_out = self.channel_attention(x)
        sa_att, sa_out = self.spatial_attention(ca_out)
        return sa_out, ca_att, sa_att

class conv_block_3d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv_3d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_3d, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UNet_CBAM_3D(nn.Module):
    def __init__(self, in_chans=1, num_classes=4):
        super(UNet_CBAM_3D, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_3d(ch_in=in_chans, ch_out=16)
        self.Conv2 = conv_block_3d(ch_in=16, ch_out=32)
        self.Conv3 = conv_block_3d(ch_in=32, ch_out=64)
        self.Conv4 = conv_block_3d(ch_in=64, ch_out=128)
        self.Conv5 = conv_block_3d(ch_in=128, ch_out=256)

        self.cbam1 = CBAM3D(channel=16)
        self.cbam2 = CBAM3D(channel=32)
        self.cbam3 = CBAM3D(channel=64)
        self.cbam4 = CBAM3D(channel=128)

        self.Up5 = up_conv_3d(ch_in=256, ch_out=128)
        self.Up_conv5 = conv_block_3d(ch_in=256, ch_out=128)

        self.Up4 = up_conv_3d(ch_in=128, ch_out=64)
        self.Up_conv4 = conv_block_3d(ch_in=128, ch_out=64)

        self.Up3 = up_conv_3d(ch_in=64, ch_out=32)
        self.Up_conv3 = conv_block_3d(ch_in=64, ch_out=32)

        self.Up2 = up_conv_3d(ch_in=32, ch_out=16)
        self.Up_conv2 = conv_block_3d(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv3d(16, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x1, ca1, sa1 = self.cbam1(x1)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2, ca2, sa2 = self.cbam2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3, ca3, sa3 = self.cbam3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4, ca4, sa4 = self.cbam4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1, (ca1, sa1, ca2, sa2, ca3, sa3, ca4, sa4)

class TwoStageBAVMSegmentation(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(TwoStageBAVMSegmentation, self).__init__()
        self.stage1_detector = UNet2D(in_chns=in_channels, class_num=1) 
        
        self.stage2_segmentor = UNet_CBAM_3D(in_chans=in_channels, num_classes=num_classes)
        
        self.roi_size = 128
        
    def calculate_mass_center(self, mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        
        coords = np.argwhere(mask > 0)
        
        if len(coords) == 0:
            return np.array(mask.shape) // 2
        
        mass_center = coords.mean(axis=0)
        return mass_center.astype(int)
    
    def extract_roi(self, volume, mass_center):
        d, h, w = volume.shape[-3:]
        roi_size = self.roi_size
        
        start_d = max(0, mass_center[0] - roi_size // 2)
        end_d = min(d, mass_center[0] + roi_size // 2)
        start_h = max(0, mass_center[1] - roi_size // 2)
        end_h = min(h, mass_center[1] + roi_size // 2)
        start_w = max(0, mass_center[2] - roi_size // 2)
        end_w = min(w, mass_center[2] + roi_size // 2)
        
        roi = volume[..., start_d:end_d, start_h:end_h, start_w:end_w]
        
        pad_d = roi_size - (end_d - start_d)
        pad_h = roi_size - (end_h - start_h)
        pad_w = roi_size - (end_w - start_w)
        
        if any([pad_d > 0, pad_h > 0, pad_w > 0]):
            roi = nn.functional.pad(roi, (0, pad_w, 0, pad_h, 0, pad_d))
        
        return roi, (start_d, end_d, start_h, end_h, start_w, end_w)
    
    def forward(self, x, phase='test'):
        if phase == 'train_stage1':
            b, c, d, h, w = x.shape
            x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(b * d, c, h, w)
            stage1_output = self.stage1_detector(x_2d)
            stage1_output = stage1_output.view(b, d, 1, h, w).permute(0, 2, 1, 3, 4)
            return stage1_output
            
        elif phase == 'train_stage2':
            return self.stage2_segmentor(x)
            
        else:  
            b, c, d, h, w = x.shape
            x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(b * d, c, h, w)
            stage1_output = self.stage1_detector(x_2d)
            stage1_mask = (torch.sigmoid(stage1_output) > 0.5).float()
            stage1_mask = stage1_mask.view(b, d, 1, h, w).permute(0, 2, 1, 3, 4)
            
            roi_coords_list = []
            roi_list = []
            
            for i in range(b):
                mass_center = self.calculate_mass_center(stage1_mask[i, 0])
                roi, coords = self.extract_roi(x[i], mass_center)
                roi_list.append(roi)
                roi_coords_list.append(coords)
            
            roi_batch = torch.stack(roi_list, dim=0)
            
            stage2_output, attentions = self.stage2_segmentor(roi_batch)
            
            full_size_output = torch.zeros_like(x[:, :1, :, :, :])
            for i in range(b):
                start_d, end_d, start_h, end_h, start_w, end_w = roi_coords_list[i]
                roi_d, roi_h, roi_w = end_d - start_d, end_h - start_h, end_w - start_w
                
                resized_output = nn.functional.interpolate(
                    stage2_output[i:i+1], 
                    size=(roi_d, roi_h, roi_w), 
                    mode='trilinear', 
                    align_corners=True
                )
                
                full_size_output[i, 0, start_d:end_d, start_h:end_h, start_w:end_w] = resized_output[0, 0]
            
            return full_size_output, stage1_mask, roi_coords_list

# 辅助函数
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def jaccard_coefficient(y_true, y_pred):
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return jaccard

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TwoStageBAVMSegmentation')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--fold', type=str)
    parser.add_argument('--lr_seg', type=float, default=0.001)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--bt_size', type=int, default=1)
    parser.add_argument('--seg_loss', type=int, default=1, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--phase', type=str, default='train_stage1', choices=['train_stage1', 'train_stage2', 'test'])
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--in_channels', type=int, default=3)

    #log_dir name
    parser.add_argument('--folder_name', type=str, default='Default_folder')

    parse_config = parser.parse_args()
    print(parse_config)
    return parse_config

def get_model(args):
    if args.model == "TwoStageBAVMSegmentation":
        model = TwoStageBAVMSegmentation(in_channels=args.in_channels, num_classes=args.num_classes)
    else:
        model = None
        print("model err")
        exit(0)
    return model.cuda()

scaler = GradScaler()

#-------------------------- train func --------------------------#
def train(epoch):
    model.train()
    iteration = 0
    optimizer.zero_grad()
    for batch_idx, batch_data in enumerate(train_loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        
        with autocast():
            if parse_config.phase == 'train_stage1':
                output = model(data, phase='train_stage1')
                loss = structure_loss(output, label) / parse_config.accumulation_steps
            else:  # train_stage2
                output, _ = model(data, phase='train_stage2')
                loss = structure_loss(output, label) / parse_config.accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % parse_config.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if (batch_idx + 1) % 10 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t[Loss: {:.4f}]'
                .format(epoch, batch_idx * len(data),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                        
        # 记录训练损失
        train_loss_writer.writerow([epoch, batch_idx, loss.item()])
        train_loss_file.flush()

#-------------------------- eval func --------------------------#
def evaluation(epoch, loader):
    model.eval()
    total_loss = 0
    dice_value = 0
    iou_value = 0
    numm = 0
    
    for batch_idx, batch_data in enumerate(loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()

        with torch.no_grad():
            if parse_config.phase == 'train_stage1':
                output = model(data, phase='train_stage1')
                loss = structure_loss(output, label)
            else:  # train_stage2 or test
                output, _ = model(data, phase='train_stage2')
                loss = structure_loss(output, label)
                
            total_loss += loss.item()
            
        output = output.sigmoid().cpu().numpy() > 0.5
        label = label.cpu().numpy()
        assert (output.shape == label.shape)
        
        dice_ave = dice_coefficient(output, label)
        iou_ave = jaccard_coefficient(output, label)

        dice_value += dice_ave
        iou_value += iou_ave
        numm += 1

    dice_average = dice_value / numm
    iou_average = iou_value / numm
    average_loss = total_loss / numm
    
    writer.add_scalar('val_metrics/val_dice', dice_average, epoch)
    writer.add_scalar('val_metrics/val_iou', iou_average, epoch)
    print("Average dice value of evaluation dataset = ", dice_average)
    print("Average iou value of evaluation dataset = ", iou_average)
    
    # 记录验证损失
    val_loss_writer.writerow([epoch, average_loss])
    val_loss_file.flush()
    
    torch.cuda.empty_cache()
    return dice_average, iou_average, average_loss

if __name__ == '__main__':
    # 打开CSV文件记录损失
    train_loss_file = open('training_loss.csv', 'w', newline='')
    train_loss_writer = csv.writer(train_loss_file)
    train_loss_writer.writerow(['Epoch', 'Batch', 'Loss'])  
    
    val_loss_file = open('validation_loss.csv', 'w', newline='')
    val_loss_writer = csv.writer(val_loss_file)
    val_loss_writer.writerow(['Epoch', 'Loss'])
    
    #-------------------------- get args --------------------------#
    parse_config = get_cfg()

    #-------------------------- build loggers and savers --------------------------#
    exp_name = parse_config.exp_name + '_phase_' + parse_config.phase + '_loss_' + str(
        parse_config.seg_loss) + '_aug_' + str(
            parse_config.aug)

    os.makedirs('logs1/{}'.format(exp_name), exist_ok=True)
    os.makedirs('logs1/{}/model'.format(exp_name), exist_ok=True)
    writer = SummaryWriter('logs1/{}/log'.format(exp_name))
    save_path = 'logs1/{}/model/best.pkl'.format(exp_name)
    latest_path = 'logs1/{}/model/latest.pkl'.format(exp_name)

    EPOCHS = parse_config.n_epochs
    os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
    device_ids = list(range(torch.cuda.device_count()))
    
    #-------------------------- build dataloaders --------------------------#
    # 注意：这里需要根据您的实际数据集类进行修改
    # 假设您的数据集类名为My3DDataset
    from src.dataloader.isbi2016_new1 import My3DDataset
    
    dataset = My3DDataset(split='train', aug=parse_config.aug)
    dataset2 = My3DDataset(split='valid', aug=False)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=parse_config.bt_size,
                                               shuffle=True,
                                               num_workers=1,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset2,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False)

    #-------------------------- build models --------------------------#
    model = get_model(parse_config)
 
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    torch.cuda.empty_cache()

    #-------------------------- start training --------------------------#
    max_dice = 0
    max_iou = 0
    best_ep = 0

    min_loss = 10
    min_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        start = time.time()
        train(epoch)
        dice, iou, loss = evaluation(epoch, val_loader)
        
        scheduler.step(loss)

        if loss < min_loss:
            min_epoch = epoch
            min_loss = loss
        else:
            if epoch - min_epoch >= parse_config.patience:
                print('Early stopping!')
                break
                
        if iou > max_iou:
            max_iou = iou
            best_ep = epoch
            torch.save(model.state_dict(), save_path)
        else:
            if epoch - best_ep >= parse_config.patience:
                print('Early stopping!')
                break
                
        torch.save(model.state_dict(), latest_path)
        time_elapsed = time.time() - start
        print(
            'Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))
            
    train_loss_file.close()
    val_loss_file.close()
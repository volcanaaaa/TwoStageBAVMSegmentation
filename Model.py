"""
Two-stage deep learning ensemble for brain arteriovenous malformations (bAVMs) nidus segmentation.
First stage uses a 2D U-Net to detect and localize bAVMs regions of interest (ROI).
Second stage employs a 3D self-attention network with CBAM for precise segmentation within the ROI.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions.uniform import Uniform

# ==================== 第一阶段：2D U-Net检测模型 ====================
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
    """Two convolution layers with batch norm and leaky relu"""

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
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock2D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock2D(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock2D(nn.Module):
    """Upsampling followed by ConvBlock"""

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

# ==================== 第二阶段：3D自注意力网络 ====================
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

# ==================== 两阶段集成模型 ====================
class TwoStageBAVMSegmentation(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(TwoStageBAVMSegmentation, self).__init__()
        # 第一阶段：2D U-Net检测模型
        self.stage1_detector = UNet2D(in_chns=in_channels, class_num=1)  # 二分类：bAVMs或背景
        
        # 第二阶段：3D自注意力网络
        self.stage2_segmentor = UNet_CBAM_3D(in_chans=in_channels, num_classes=num_classes)
        
        # ROI大小
        self.roi_size = 128
        
    def calculate_mass_center(self, mask):
        """计算二值掩码的质量中心"""
        # 将掩码转换为numpy数组
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        
        # 找到非零元素的坐标
        coords = np.argwhere(mask > 0)
        
        if len(coords) == 0:
            # 如果没有检测到任何bAVMs，返回图像中心
            return np.array(mask.shape) // 2
        
        # 计算质量中心
        mass_center = coords.mean(axis=0)
        return mass_center.astype(int)
    
    def extract_roi(self, volume, mass_center):
        """根据质量中心提取ROI"""
        d, h, w = volume.shape[-3:]
        roi_size = self.roi_size
        
        # 计算ROI的起始和结束坐标
        start_d = max(0, mass_center[0] - roi_size // 2)
        end_d = min(d, mass_center[0] + roi_size // 2)
        start_h = max(0, mass_center[1] - roi_size // 2)
        end_h = min(h, mass_center[1] + roi_size // 2)
        start_w = max(0, mass_center[2] - roi_size // 2)
        end_w = min(w, mass_center[2] + roi_size // 2)
        
        # 提取ROI
        roi = volume[..., start_d:end_d, start_h:end_h, start_w:end_w]
        
        # 如果需要，进行填充以确保ROI大小一致
        pad_d = roi_size - (end_d - start_d)
        pad_h = roi_size - (end_h - start_h)
        pad_w = roi_size - (end_w - start_w)
        
        if any([pad_d > 0, pad_h > 0, pad_w > 0]):
            roi = nn.functional.pad(roi, (0, pad_w, 0, pad_h, 0, pad_d))
        
        return roi, (start_d, end_d, start_h, end_h, start_w, end_w)
    
    def forward(self, x, phase='test'):
        """
        前向传播
        
        参数:
        - x: 输入的多模态图像 [B, C, D, H, W]
        - phase: 训练阶段 ('train_stage1', 'train_stage2') 或测试阶段 ('test')
        """
        if phase == 'train_stage1':
            # 第一阶段训练：处理2D切片
            b, c, d, h, w = x.shape
            # 将3D体积重塑为2D切片批次
            x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(b * d, c, h, w)
            # 通过第一阶段网络
            stage1_output = self.stage1_detector(x_2d)
            # 重塑回3D形状
            stage1_output = stage1_output.view(b, d, 1, h, w).permute(0, 2, 1, 3, 4)
            return stage1_output
            
        elif phase == 'train_stage2':
            # 第二阶段训练：直接使用裁剪的ROI
            return self.stage2_segmentor(x)
            
        else:  # 测试阶段
            # 第一阶段：检测bAVMs位置
            b, c, d, h, w = x.shape
            x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(b * d, c, h, w)
            stage1_output = self.stage1_detector(x_2d)
            stage1_mask = (torch.sigmoid(stage1_output) > 0.5).float()
            stage1_mask = stage1_mask.view(b, d, 1, h, w).permute(0, 2, 1, 3, 4)
            
            # 计算质量中心并提取ROI
            roi_coords_list = []
            roi_list = []
            
            for i in range(b):
                mass_center = self.calculate_mass_center(stage1_mask[i, 0])
                roi, coords = self.extract_roi(x[i], mass_center)
                roi_list.append(roi)
                roi_coords_list.append(coords)
            
            roi_batch = torch.stack(roi_list, dim=0)
            
            # 第二阶段：在ROI内进行精细分割
            stage2_output, attentions = self.stage2_segmentor(roi_batch)
            
            # 将分割结果映射回原始图像空间
            full_size_output = torch.zeros_like(x[:, :1, :, :, :])
            for i in range(b):
                start_d, end_d, start_h, end_h, start_w, end_w = roi_coords_list[i]
                roi_d, roi_h, roi_w = end_d - start_d, end_h - start_h, end_w - start_w
                
                # 调整分割结果大小以匹配原始ROI大小
                resized_output = nn.functional.interpolate(
                    stage2_output[i:i+1], 
                    size=(roi_d, roi_h, roi_w), 
                    mode='trilinear', 
                    align_corners=True
                )
                
                # 将结果放回原始图像中的正确位置
                full_size_output[i, 0, start_d:end_d, start_h:end_h, start_w:end_w] = resized_output[0, 0]
            
            return full_size_output, stage1_mask, roi_coords_list

# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 创建模型
    model = TwoStageBAVMSegmentation(in_channels=3, num_classes=4)
    
    # 测试第一阶段
    print("Testing stage 1...")
    test_input = torch.rand(1, 3, 32, 256, 256)  # 模拟多模态3D输入
    stage1_output = model(test_input, phase='train_stage1')
    print(f"Stage 1 output shape: {stage1_output.shape}")
    
    # 测试第二阶段
    print("Testing stage 2...")
    roi_input = torch.rand(1, 3, 128, 128, 128)  # 模拟ROI输入
    stage2_output, attentions = model(roi_input, phase='train_stage2')
    print(f"Stage 2 output shape: {stage2_output.shape}")
    
    # 测试完整流程
    print("Testing full pipeline...")
    full_output, coarse_mask, roi_coords = model(test_input, phase='test')
    print(f"Full output shape: {full_output.shape}")
    print(f"Coarse mask shape: {coarse_mask.shape}")
    print(f"ROI coordinates: {roi_coords}")
import os
import argparse
import torch
from torch.utils.data import DataLoader
import time
import sys
import numpy as np
original_path = sys.path.copy()
sys.path.append('../')#cause

# 添加TwoStageBAVMSegmentation模型导入
from networks.TwoStageBAVMSegmentation import TwoStageBAVMSegmentation
from src.dataloader.isbi2016_new1 import My3DDataset  # 假设这是您的3D数据集类

sys.path = original_path
import torchvision
#from utils.isbi2016_new import norm01, myDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="TwoStageBAVMSegmentation",
                    choices=["TwoStageBAVMSegmentation"], help='model')
parser.add_argument('--base_dir', type=str, default="/fs0/home/sz2106159/jpg_seg/jpg_seg/datasets", help='dir')
parser.add_argument('--train_file_dir', type=str, default="busi_train_all", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val_all", help='dir')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=1,  # 对于3D模型，batch_size通常设为1
                    help='batch_size per gpu')
parser.add_argument('--epochs', type=int, default=300,
                    help='')
parser.add_argument('--weights', type=str, default=r'/fs0/home/sz2106159/jpg_seg/jpg_seg/src/logs/test_loss_1_aug_1/model/latest.pkl',
                    help='')
parser.add_argument('--mode', type=str, default=r'test',
                    help='')# optional['val','test']
parser.add_argument('--time', type=bool, default=False,
                    help='如果计算时间就不保存预测和标签')
parser.add_argument('--in_channels', type=int, default=3,
                    help='输入通道数')
parser.add_argument('--num_classes', type=int, default=1,
                    help='输出类别数')
args = parser.parse_args()

def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    return iou, dice

def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def remove_para(weights):
    from collections import OrderedDict
    new = OrderedDict()
    for K,V in weights.items():
        name = K[7:]
        new [name] = V
    return new

def get_model(args):
    if args.model == "TwoStageBAVMSegmentation":
        model = TwoStageBAVMSegmentation(in_channels=args.in_channels, num_classes=args.num_classes)
    else:
        model = None
        print("model err")
        exit(0)
    try:
        model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
    except:
        model.load_state_dict(remove_para(torch.load(args.weights, map_location=torch.device('cpu'))))
    return model.cuda()

def binarize_tensor(tensor):
    """
    将张量二值化，阈值取平均值
    Args:
        tensor: 输入的张量

    Returns:
        二值化后的张量
    """
    threshold = tensor.mean()  # 取张量的平均值作为阈值
    device = tensor.device
    binary_tensor = torch.where(tensor > threshold, torch.tensor(1,device=device), torch.tensor(0,device=device))
    return binary_tensor

def save_3d_volume_as_slices(volume, save_dir, name_prefix):
    """
    将3D体积保存为2D切片
    Args:
        volume: 3D张量 (D, H, W)
        save_dir: 保存目录
        name_prefix: 文件名前缀
    """
    os.makedirs(save_dir, exist_ok=True)
    for d in range(volume.shape[0]):
        slice_2d = volume[d]
        # 转换为PIL图像并保存
        slice_image = torchvision.transforms.ToPILImage()(slice_2d.cpu().numpy().astype('uint8'))
        slice_path = os.path.join(save_dir, f'{name_prefix}_slice_{d:03d}.png')
        slice_image.save(slice_path)

def inference(args, fold):
    if args.mode == 'test':  # infer test imgs
        dataset = My3DDataset(split='test', aug=False)
    elif args.mode == 'val':  # infer val imgs
        dataset = My3DDataset(split='valid', aug=False)
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    model = get_model(args)
    model.eval()
    num = 0
    iou_total = 0
    dice_total = 0

    save_dir = 'test1_predicted_segmentation_volumes'
    save_dir_label = 'test1_label_segmentation_volumes'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_label, exist_ok=True)
    
    start = time.time()
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_loader):
            input, target = sampled_batch['image'], sampled_batch['label']
            name = sampled_batch['idx']
            name = str(name)
            name = name.split("/")[-1].replace("']", '')
            print(f"Processing: {name}")

            input = input.cuda()
            target = target.cuda()

            # 使用TwoStageBAVMSegmentation模型进行推理
            output, stage1_mask, roi_coords_list = model(input, phase='test')

            iou, dice = iou_score(output, target)
            num += 1
            iou_total += iou
            dice_total += dice

            # 处理输出
            output = torch.sigmoid(output)
            output_binary = binarize_tensor(output) * 255.0

            if not args.time:
                # 保存预测结果
                for i in range(output_binary.shape[0]):  # 遍历batch
                    volume_pred = output_binary[i, 0]  # 取第一个通道 (D, H, W)
                    save_3d_volume_as_slices(volume_pred, save_dir, f'{name}_pred_{i}')
                
                # 保存标签
                for i in range(target.shape[0]):  # 遍历batch
                    volume_label = target[i, 0] * 255.0  # 取第一个通道并缩放到0-255
                    save_3d_volume_as_slices(volume_label, save_dir_label, f'{name}_label_{i}')

    print("IoU: ", iou_total / num)
    print("DSC: ", dice_total / num)
    print("Total Inference Time: ", time.time() - start)
    print("Avg Inference Time: ", (time.time() - start) / num)

if __name__ == "__main__":
    inference(args, fold=5)  # fold5=======>train val:8 2
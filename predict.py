import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import efficientnet_b3


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.SiLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out


class UBlockSet(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True):
        super().__init__()
        self.block1 = UBlock(in_channels, out_channels)
        self.block2 = UBlock(out_channels, out_channels)
        self.residual = residual
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(0.03)
        
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        out = self.block2(self.block1(x))
        if self.residual:
            if self.downsample:
                x = self.downsample(x)
            out += x
        out = self.se(out)
        out = self.dropout(out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = UBlock(in_channels, out_channels)
        
    def forward(self, x):
        size = (x.shape[-2] * 2, x.shape[-1] * 2)
        out = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
        out = self.block(out)
        return out


class ChannelReductionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        out = self.act(self.bn(self.conv(x)))
        return out


class Interpolate(nn.Module):
    def __init__(self, size, mode="bicubic", align_corners=False, antialias=False):
        super().__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
        self.antialias = antialias

    def forward(self, x):
        result = F.interpolate(x, 
                               size=self.size, 
                               mode=self.mode, 
                               align_corners=self.align_corners, 
                               antialias=self.antialias)
        return result


class FullScaleConnection(nn.Module):
    def __init__(self, connection_indx, out_channels, encoder_channels, reduced_channels=24, img_size=512):
        super().__init__()
        output_size = 512 // 2**(5 - connection_indx)
        self.upsample_count = connection_indx
        self.downsample_count = 4 - connection_indx
        self.reduces = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, reduced_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduced_channels),
                nn.SiLU()
            ) for ch in encoder_channels
        ])
        self.upsample = Interpolate(size=output_size, antialias=False)
        self.downsample = Interpolate(size=output_size, antialias=True)
        self.channel_reduction = ChannelReductionBlock(reduced_channels * 5, out_channels)
        

    def forward(self, x):
        result = [] 
        for out, reduce in zip(x, self.reduces):
            result.append(reduce(out))
        for i in range(self.upsample_count):
            result[i] = self.upsample(result[i])
        for i in range(self.downsample_count):
            result[4-i] = self.downsample(result[4-i])
        out = self.channel_reduction(result)
        return out


class SPPF(nn.Module):
    def __init__(self, channels, k=5):
        super().__init__()
        c = max(16, channels // 2)
        self.cv1 = UBlock(channels, c, kernel_size=1, padding=0)
        self.cv2 = UBlock(c * 4, channels, kernel_size=1, padding=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
    def forward(self, x):
        out = self.cv1(x)
        tensors_to_concat = [out]
        
        for _ in range(3):
            out = self.m(out)
            tensors_to_concat.append(out)
        
        concat_tensor = torch.cat(tensors_to_concat, dim=1)
        out = self.cv2(concat_tensor)
        return x + out


class ResUNet(nn.Module):
    def __init__(self, classes=2, img_size=512):
        super().__init__()
        self.channels = [32, 24, 32, 48, 136, 384, 512]  
        backbone = efficientnet_b3(weights='DEFAULT').features
        self.block0 = nn.Sequential(
            backbone[0],
            backbone[1]
        )
        self.block1 = backbone[2]
        self.block2 = backbone[3]
        self.block3 = nn.Sequential(
            backbone[4],
            backbone[5]
        )
        self.block4 = nn.Sequential(
            backbone[6],
            backbone[7]
        )
        self.sppf = SPPF(self.channels[-2])
        self.final_block = nn.Sequential(
            nn.Conv2d(self.channels[-2], self.channels[-1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[-1]),
            nn.SiLU(),
            nn.Dropout2d(0.05),
            UBlockSet(self.channels[-1], self.channels[-1]),
            UpsampleBlock(self.channels[-1], self.channels[-2])
        )

        encoder_channels = self.channels[-2:0:-1]
        for i in range(len(self.channels)-2, 0, -1):
            full_scale_block = FullScaleConnection(5 - i, self.channels[i], encoder_channels)
            skip_block = UBlockSet(self.channels[i], self.channels[i])
            channel_reduction = ChannelReductionBlock(self.channels[i] * 2, self.channels[i])
            block = UBlockSet(self.channels[i], self.channels[i-1])
            upsample = UpsampleBlock(self.channels[i-1], self.channels[i-1])
            indx = len(self.channels)-2
            setattr(self, f"full_scale_block{indx-i}", full_scale_block)
            setattr(self, f"skip_block{indx-i}", skip_block)
            setattr(self, f"channel_reduction{indx-i}", channel_reduction)
            setattr(self, f"upblock{indx-i}", block)
            setattr(self, f"upsample{indx-i}", upsample)

        self.fconv = nn.Sequential(
            UBlockSet(self.channels[0], self.channels[0]),
            UBlock(self.channels[0], self.channels[0]),
            nn.Conv2d(self.channels[0], classes, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # энкодер
        out1 = self.block0(x)
        out2 = self.block1(out1)
        out3 = self.block2(out2)
        out4 = self.block3(out3)
        out5 = self.block4(out4)
        out = self.sppf(out5)
        out = self.final_block(out)
        encoder_outs = [out5, out4, out3, out2, out1]
        
        # декодер
        full_scale_out = self.full_scale_block0(encoder_outs)
        skip_out = self.skip_block0(full_scale_out)
        out = self.channel_reduction0([out, skip_out])
        out = self.upblock0(out)
        out = self.upsample0(out)

        full_scale_out = self.full_scale_block1(encoder_outs)
        skip_out = self.skip_block1(full_scale_out)
        out = self.channel_reduction1([out, skip_out])
        out = self.upblock1(out)
        out = self.upsample1(out)

        full_scale_out = self.full_scale_block2(encoder_outs)
        skip_out = self.skip_block2(full_scale_out)
        out = self.channel_reduction2([out, skip_out])
        out = self.upblock2(out)
        out = self.upsample2(out)

        full_scale_out = self.full_scale_block3(encoder_outs)
        skip_out = self.skip_block3(full_scale_out)
        out = self.channel_reduction3([out, skip_out])
        out = self.upblock3(out)
        out = self.upsample3(out)

        full_scale_out = self.full_scale_block4(encoder_outs)
        skip_out = self.skip_block4(full_scale_out)
        out = self.channel_reduction4([out, skip_out])
        out = self.upblock4(out)
        out = self.upsample4(out)

        # классификация пикселей
        out = self.fconv(out)
        return out


def predict(args):
    """
    Основная функция для выполнения инференса.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResUNet(classes=2, img_size=512)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    # Трансформации для подачи в модель
    model_transforms = A.Compose([
        A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_LANCZOS4),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, p=1),
        A.Normalize(),
        ToTensorV2(),
    ])

    image_original = cv2.imread(args.image)
    if image_original is None:
        raise FileNotFoundError(f"Image not found at {args.image}")
    
    original_h, original_w = image_original.shape[:2]
    image_original_rgb = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    
    # Применяем трансформации и делаем предсказание
    image_transformed_for_model = model_transforms(image=image_original_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_transformed_for_model)
    mask_512 = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    scale = 512 / max(original_h, original_w)
    resized_h = int(original_h * scale)
    resized_w = int(original_w * scale)

    # Вычисляем, сколько полей (паддинга) было добавлено
    pad_top = (512 - resized_h) // 2
    pad_left = (512 - resized_w) // 2

    mask_unpadded = mask_512[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]
    final_mask = cv2.resize(mask_unpadded, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # Создаем 4-канальное изображение (BGRA) из оригинального изображения
    result_rgba = cv2.cvtColor(image_original, cv2.COLOR_BGR2BGRA)
    result_rgba[:, :, 3] = final_mask * 255

    cv2.imwrite(args.output, result_rgba)
    print(f"Result saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Инференс модели сегментации для удаления фона.")
    parser.add_argument('--image', type=str, required=True, help='Путь к входному изображению.')
    parser.add_argument('--weights', type=str, required=True, help='Путь к файлу с весами модели (.pth).')
    parser.add_argument('--output', type=str, required=True, help='Путь для сохранения результата (рекомендуется .png).')
    
    args = parser.parse_args()
    predict(args)
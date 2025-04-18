# model.py
import torch
import torch.nn as nn
import common # 导入共享变量

class MyModel(nn.Module): # 类名建议用大写开头
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义卷积块创建函数，避免重复代码
        def create_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2) # 明确指定 kernel_size 和 stride
            )

        # 卷积层: (Batch, Channels, Height, Width)
        # Input: (B, 1, 60, 160)
        self.layer1 = create_conv_block(1, 64)      # Output: (B, 64, 30, 80)
        self.layer2 = create_conv_block(64, 128)     # Output: (B, 128, 15, 40)
        self.layer3 = create_conv_block(128, 256)    # Output: (B, 256, 7, 20) - 注意这里的尺寸变化
        self.layer4 = create_conv_block(256, 512)    # Output: (B, 512, 3, 10) - 注意这里的尺寸变化

        # 计算进入全连接层前的扁平化特征数量
        # 512 (channels) * 3 (height) * 10 (width) = 15360
        self.flattened_size = 512 * 3 * 10

        # 全连接层 (分类器部分)
        self.layer6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 4096),
            nn.Dropout(0.2), # Dropout 位置可以在 Linear 后或 ReLU 后
            nn.ReLU(),
            # 输出层: 输出维度 = 验证码长度 * 字符集大小
            nn.Linear(4096, common.captcha_size * common.num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer6(x)
        # 输出形状是 (Batch, captcha_size * num_classes)
        # 在训练/预测时需要 reshape 成 (Batch, captcha_size, num_classes)
        return x

# 测试模型结构 (可选)
if __name__ == '__main__':
    # 创建一个模拟的输入数据 (batch_size=64, channels=1, height=60, width=160)
    data = torch.randn(64, 1, common.image_height, common.image_width).to(common.device)
    model = MyModel().to(common.device)
    output = model(data)
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output.shape}")
    # 验证输出维度是否正确
    expected_output_size = 64 * common.captcha_size * common.num_classes
    print(f"Expected output elements per batch item: {common.captcha_size * common.num_classes}")
    print(f"Actual output elements: {output.numel()}")
    assert output.shape == (64, common.captcha_size * common.num_classes), "Output shape mismatch!"
    print("Model definition test passed.")
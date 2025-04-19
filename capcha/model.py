import torch
import torch.nn as nn
import capcha.common as common

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        def create_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # Input: (B, 1, 60, 160)
        self.layer1 = create_conv_block(1, 64)      # Output: (B, 64, 30, 80)
        self.layer2 = create_conv_block(64, 128)    # Output: (B, 128, 15, 40)
        self.layer3 = create_conv_block(128, 256)   # Output: (B, 256, 7, 20)
        self.layer4 = create_conv_block(256, 512)   # Output: (B, 512, 3, 10)

        # 512 (channels) * 3 (height) * 10 (width) = 15360
        self.flattened_size = 512 * 3 * 10

        self.layer6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 4096),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(4096, common.captcha_size * common.num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer6(x)
        return x

if __name__ == '__main__':
    data = torch.randn(64, 1, common.image_height, common.image_width).to(common.device)
    model = MyModel().to(common.device)
    output = model(data)
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output.shape}")
    expected_output_size = 64 * common.captcha_size * common.num_classes
    print(f"Expected output elements per batch item: {common.captcha_size * common.num_classes}")
    print(f"Actual output elements: {output.numel()}")
    assert output.shape == (64, common.captcha_size * common.num_classes), "Output shape mismatch!"
    print("Model definition test passed.")
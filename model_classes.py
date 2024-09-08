from torch import nn

class ConvolutionalNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.convolutional_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.flatten = nn.Flatten()
        self.dense_stack = nn.Sequential(
            nn.Linear(256*3*3, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.model_chunks = [self.convolutional_stack, self.flatten, self.dense_stack]
    def forward(self, x):
        x = self.convolutional_stack(x)
        x = self.flatten(x)
        logits = self.dense_stack(x)
        return logits
    
class BiggerConvolutionalNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.convolutional_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.dense_stack = nn.Sequential(
            nn.Linear(1024*3*3, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes),
        )

        self.model_chunks = [self.convolutional_stack, self.flatten, self.dense_stack]
    def forward(self, x):
        x = self.convolutional_stack(x)
        x = self.flatten(x)
        logits = self.dense_stack(x)
        return logits
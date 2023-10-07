import torch.nn as nn
import torch.nn.functional as F

class IEGMNet(nn.Module):
    def __init__(self):
        super(IEGMNet, self).__init__()
        self.norm = nn.Sequential(
            nn.BatchNorm1d(1, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.skip1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=11, stride=4, padding=0),
        )

        self.skip2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=20, kernel_size=7, stride=4, padding=0),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=6, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=5, kernel_size=5, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=10, kernel_size=4, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=4, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=4, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=740, out_features=10)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, input):

        norm_output = self.norm(input)
        conv1_output = self.conv1(norm_output)
        # conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output) + self.skip1(norm_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output) + self.skip2(conv2_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(-1,740)

        fc1_output = F.relu(self.fc1(conv5_output))
        fc2_output = self.fc2(fc1_output)
        return fc2_output
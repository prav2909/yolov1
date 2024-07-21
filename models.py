import torch
import torch.nn as nn

class YoloV1Model(nn.Module):
	def __init__(self, S, B, C):
		super(YoloV1Model, self).__init__()
		self.S = S       # Divide each image into a SxS grid
		self.B = B       # Number of bounding boxes to predict
		self.C = C      # Number of classes in the dataset
		self.depth = self.B * 5 + self.C 
		block_1 = [
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), 
			nn.BatchNorm2d(64),
			nn.LeakyReLU(negative_slope=0.1),
			nn.MaxPool2d(kernel_size=2, stride=2),
		]
		block_2 = [

			nn.Conv2d(64, 192, kernel_size=3, padding=1), 
			nn.BatchNorm2d(192),
			nn.LeakyReLU(negative_slope=0.1),
			nn.MaxPool2d(kernel_size=2, stride=2),
		]
		block_3 = [
			nn.Conv2d(192, 128, kernel_size=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(negative_slope=0.1),
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(negative_slope=0.1),
			nn.Conv2d(256, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(negative_slope=0.1),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(negative_slope=0.1),
			nn.MaxPool2d(kernel_size=2, stride=2),
		]
		block_4 = [
			nn.Conv2d(512, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(negative_slope=0.1),
			nn.Conv2d(512, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(negative_slope=0.1),
			nn.Conv2d(512, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(negative_slope=0.1),
			nn.Conv2d(512, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.LeakyReLU(negative_slope=0.1),
		]
		block_5 = [
			nn.Conv2d(512, 512, kernel_size=1),
			nn.LeakyReLU(negative_slope=0.1),
			nn.BatchNorm2d(512),
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(negative_slope=0.1),
			nn.MaxPool2d(kernel_size=2, stride=2),
		]
		block_6 = [
			nn.Conv2d(1024, 512, kernel_size=1),
			nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
			nn.Conv2d(1024, 512, kernel_size=1),
			nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
		]
		block_7 = [
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
		]
		block_8 = [
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
		]
		block_9 = [
			nn.Flatten(),
			nn.Linear(self.S * self.S * 1024, 4096),
			nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, self.S * self.S * self.depth), 
			nn.Sigmoid()
		]
		# self.model = nn.Sequential(*block_1, Print("block_1"), 
		# 					 *block_2, Print("block_2"), 
		# 					 *block_3, Print("block_3"),
		# 					 *block_4, Print("block_4"),
		# 					 *block_5, Print("block_5"),
		# 					 *block_6, Print("block_6"),
		# 					 *block_7, Print("block_7"),
		# 					 *block_8, Print("block_8"),
		# 					 *block_9, Print("block_9"))
		self.model = nn.Sequential(*block_1, 
							 *block_2, 
							 *block_3,
							 *block_4,
							 *block_5,
							 *block_6,
							 *block_7,
							 *block_8,
							 *block_9)
	def forward(self, x):
		out = self.model.forward(x)
		return torch.reshape(
            out,
            (x.size(dim=0), self.S, self.S, self.depth)
        )
		
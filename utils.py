import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from torchvision.models import resnet50, ResNet50_Weights

def calculate_L2_norm(model):
	"""
	Calculates L2 Norm
	"""
	total_norm = 0.0
	for param in model.parameters():
		if param.requires_grad:
			param_norm = torch.norm(param.data, 2)
			total_norm += param_norm.item() ** 2
	return total_norm

class Print(nn.Module):
	"""
	Prints shape of the output of previous layer when this Print is called in a sequential stack.
	"""
	def __init__(self, msg_to_disp=""):
		super(Print, self).__init__()
		self.msg_to_disp = msg_to_disp
	def forward(self, x):
		if(self.msg_to_disp == "block_9"):
			t=1
			pass
		print(self.msg_to_disp + " "+ str(x.shape))
		return x

def parse_kitti_bbox(label_path, class_mapping, scale_x, scale_y):
	"""
	This function parses bounding box data from KITTI label files. 
	It reads the label text files provided by the KITTI database, where each line represents an object in an image. 
	The function extracts the top-left and bottom-right coordinates of each bounding box, as well as the object class. 
	Afterwards, it calculates the centre of BB.
	The extracted information is then returned as a list.
	"""
	with open(label_path, 'r') as f:
		# Each line corresponds to one object
		ret_data = []
		for line in f:
			data = line.strip().split(' ')
			# Assuming data format: type, truncated, occluded, ... , xmin, ymin, xmax, ymax
			if (data[0] not in ['DontCare', 'Misc', 'Tram']) and (data[1] != 1) and (data[2] not in (2,3)):  # Skip some objects
				obj_class = class_mapping[data[0]]
				x_left_top = float(data[4])*scale_x
				y_left_top = float(data[5])*scale_y
				x_right_bottom = float(data[6])*scale_x
				y_right_bottom = float(data[7])*scale_y

				# Now calculate centre cordinates and w, h
				width = x_right_bottom - x_left_top
				height = y_right_bottom - y_left_top
				x_centre = x_left_top + width/2
				y_centre = y_left_top + height/2

				ret_data.append([obj_class, x_centre, y_centre, width, height])
		return np.array(ret_data)
	return None  # No bounding box found

def get_files_in_folder(folder_path, ftype=None):
	if ftype is None:
		return [os.path.abspath(os.path.join(folder_path, file)) for file in os.listdir(folder_path)]
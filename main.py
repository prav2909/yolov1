"""
Author: Praveen Kumar
This script serves as the entry point for training a yolov1 model on the KITTI Dataset. 
I have intentionally kept the model's tone simple to enhance readability. 
Although some optimization opportunities might be apparent, I made deliberate choices to prioritize understanding.

For detailed instructions on utilizing this script, kindly refer to the readme.md file. 
It provides comprehensive information to facilitate your usage of the script effectively.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import math
from torchviz import make_dot

from loss_fn import Loss

from utils import calculate_L2_norm
from kitti_dataset import KITTIImageLoader
from models import YoloV1Model
# from utils_ import plot_bounding_box_with_yolo_compatible_bb_dims

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(
		data_dir,   # Directory path where data is stored
		BATCH_SIZE,   # Batch size for training
		SHUFFLE,   # Shuffle the data during training
		NUM_OF_EPOCHS,   # Number of epochs for training
		S,   # Number of grid cells along each image dimension
		B,   # Number of bounding boxes per grid cell
		C,   # Number of classes
		new_img_size,   # New size for resizing images
		class_mapping   # Mapping of class labels to integers
	):
	
	KITTIDataset = KITTIImageLoader(data_dir, class_mapping, S, B, C, new_img_size)
	data_loader = DataLoader(KITTIDataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

	# Create an instance of the YoloV1 model and move it to the device (e.g., GPU)
	yoloV1 = YoloV1Model(S, B, C).to(device)

	# Initialize model weights with Xavier initialization to avoid exploding gradients
	for layer in yoloV1.modules():
		if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
			# Use Xavier Uniform initialization
			nn.init.xavier_uniform_(layer.weight)
			nn.init.constant_(layer.bias, 0.0)  # Initialize biases to 0.0
	
	# Define the loss function and move it to the device
	criterion = Loss(feature_size=S, num_bboxes=B, num_classes=len(class_mapping)).to(device)

	# Define the optimizer
	optimizer = optim.Adam(yoloV1.parameters(), lr=0.0001, weight_decay=1)

	total_loss = 0
	
	loss_list = []
	
	# Training loop
	for epoch in range(NUM_OF_EPOCHS):
		for step, a_dict in enumerate(data_loader):
			input = (a_dict['img'].float()).to(device)   # Get the input images
			gt = (a_dict['label'].float()).to(device)   # Get the ground truth labels
			optimizer.zero_grad()   # Zero out the gradients
			pred = yoloV1.forward(input)   # Forward pass to obtain predictions
			if torch.isnan(pred).any():
				pred = yoloV1.forward(input)
			loss = criterion(pred, gt)   # Calculate the loss
			loss.backward()   # Backward pass to compute gradients
			optimizer.step()   # Update model parameters
			total_loss += loss.item()
			if step%100 == 0:
				loss_list.append(loss.item())
			if step%10 == 0:
				total_norm = calculate_L2_norm(yoloV1)  # Calculate the L2 norm of the model weights
				print("Epoch: " +str(epoch) + "; loss.item(): " +str(loss.item()) + "; Total Norm: " + str(total_norm))
		print(f"Epoch [{epoch + 1}/{NUM_OF_EPOCHS}], Loss: {total_loss/len(data_loader):.4f}")
	
	# Save the trained model
	torch.save(yoloV1.state_dict(), './yoloV1.pth')

def main():
	# Define the directory path for the dataset
	data_dir = 'D:/KITTI_DS/'

	# Hyperparameters for training
	BATCH_SIZE = 16  # Batch size for training
	SHUFFLE = True  # Whether to shuffle the dataset
	NUM_OF_EPOCHS = 20  # Number of epochs for training

	# Hyperparameters for YOLO model
	S = 7  # Grid size
	B = 5  # Number of bounding boxes
	C = 7  # Number of classes

	# Image size for resizing
	new_img_size = 448  # New size of image

	# Class mapping for object detection
	class_mapping = {
		'Car': 0,
		'Cyclist': 1,
		'Misc': 2,
		'Pedestrian': 3,
		'Person_sitting': 3,
		'Truck': 4,
		'Van': 5
	}
	train(
		data_dir,
		BATCH_SIZE,
		SHUFFLE,
		NUM_OF_EPOCHS,
		S,
		B,
		C,
		new_img_size,
		class_mapping
		)

	#TODO: Extend test and inference


if __name__ == "__main__":
	main()

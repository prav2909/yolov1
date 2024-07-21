from torch.utils.data import Dataset, DataLoader
import cv2
import os
import math
import numpy as np

from utils import get_files_in_folder, parse_kitti_bbox

class KITTIImageLoader(Dataset):
	# Constructor for the KITTIImageLoader class
	def __init__(self, data_dir, class_mapping, S, B, C, new_img_size=448, train_split=0.7, test_split=0.3):
		self.S = S       # Divide each image into a SxS grid
		self.B = B       # Number of bounding boxes to predict
		self.C = C      # Number of classes in the dataset
		self.grid_scale = new_img_size/self.S 
		self.depth = self.B * 5 + self.C 
		self.data_dir = data_dir
		self.new_size = new_img_size
		self.train_img_path = self.data_dir + '/training_images/' # Path where images are stored
		self.train_label_path = self.data_dir + '/training_labels/' # Path where labels for images are stores in form of text files
		self.class_mapping = class_mapping
		
        # Check if the required directories exist
		if not os.path.isdir(self.data_dir) or not os.path.isdir(self.train_img_path) or not os.path.isdir(
				self.train_label_path):
			raise ValueError("Not a directory: {self.data_dir}")
		
        # Get the list of all image and label files in the training directories
		self.list_of_all_imgs = get_files_in_folder(self.train_img_path)
		self.list_of_all_labels = get_files_in_folder(self.train_label_path)
		
        # Check if the number of image files matches the number of label files
		if len(self.list_of_all_imgs) != len(self.list_of_all_labels):
			raise ValueError("Number of images and labels do not match")
		pass
	
        # Randomly select a subset of images and labels for training
		ind_for_train = np.random.randint(0, len(self.list_of_all_imgs), int(train_split*len(self.list_of_all_imgs)))
		self.list_of_train_imgs = [self.list_of_all_imgs[i] for i in ind_for_train]
		self.list_of_train_labels = [self.list_of_all_labels[i] for i in ind_for_train]
	# Returns the total number of training images
	def __len__(self):
		return len(self.list_of_train_imgs)

    # Returns a dictionary containing the resized image and its ground truth label
	def __getitem__(self, index):
		# Initialize the ground truth tensor with zeros
		ground_truth = np.zeros((self.S, self.S, self.depth))
		
        # Read and resize the image
		img = cv2.imread(self.list_of_train_imgs[index])
		resized_img = cv2.resize(img, (self.new_size, self.new_size))
		resized_img = np.transpose(resized_img, (2,0,1))
		scale_x = self.new_size / img.shape[1] #NEW_SIZE / original_width
		scale_y = self.new_size / img.shape[0] #NEW_SIZE / original_height
		
        # Parse the bounding box information from the label file
		objClass_and_bb_from_kitti_text_label_files = parse_kitti_bbox(self.list_of_train_labels[index], self.class_mapping, scale_x, scale_y)
		
        # Process each bounding box and update the ground truth tensor
		for iter in range(len(objClass_and_bb_from_kitti_text_label_files)):
			[obj_class, x_centre, y_centre, width, height] = objClass_and_bb_from_kitti_text_label_files[iter]
			# plot_bounding_box_with_yolo_compatible_bb_dims(resized_img.transpose((1,2,0)), x_centre, y_centre, width, height)
			
			# Create a one-hot encoding for the object class
			one_hot_obj_class = np.zeros((self.C))
			one_hot_obj_class[int(obj_class)] = 1
			
            # Calculate the grid indices where the object's center lies
			x_ind = math.ceil(x_centre/self.grid_scale) - 1
			y_ind = math.ceil(y_centre/self.grid_scale) - 1
			
            # Normalize the width and height of the bounding box w.r.t image
			width /= self.new_size
			height /= self.new_size

			"""
			Important observation here: As you can see in the below for-loop, for one grid, all bounding boxes have same information,
			This is because the original YoloV1 only predicted one objevct per grid. This was the main
			disadvantage of yoloV1.
			"""
			for k in range(self.B):
				s = 5 * k
				ground_truth[x_ind, y_ind, s] = (x_centre % self.grid_scale) / self.grid_scale 
				ground_truth[x_ind, y_ind, s] = (y_centre % self.grid_scale ) / self.grid_scale
				ground_truth[x_ind, y_ind, s+2] = width
				ground_truth[x_ind, y_ind, s+3] = height
				ground_truth[x_ind, y_ind, s+4] = 1.0 # Is Object present
				ground_truth[x_ind, y_ind, s+5:s+5+self.C] = one_hot_obj_class
        # Create and return a dictionary containing the resized image and its ground truth label
		ret_dict = {'img': resized_img, 'label': np.array(ground_truth)}

		return ret_dict
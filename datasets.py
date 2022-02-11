import torch
import cv2
import numpy as np
import os
import glob as glob
from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform, get_train_transform_without_boxes
import json

# the dataset class
class MicrocontrollerDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None, transforms_without_boxes=None):
        self.transforms_without_boxes = transforms_without_boxes
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        
        # get all the image paths in sorted order    
        self.image_paths = glob.glob(f"{self.dir_path}/*.png")
        self.all_images = [image_path.split('/')[-1].split('.')[0] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

        # FileNames Dictionary
        self.filenames = dict()
        for sub_dir, dir_name, files in os.walk("data/out_bbox/"):
            for file in files:
                try:
                    with open("data/out_bbox/" + file, 'r') as fp:
                        file = file.split('.')[0]
                        self.filenames[file] = json.load(fp)
                except Exception as e:
                    pass
                
    def check_bounding_boxes(self, box_coordinates):
        box_coords = []
        for coordinate in box_coordinates:
            if coordinate < 1:
                box_coords.append(1)
            elif coordinate > self.width:
                box_coords.append(self.width)
            else:
                box_coords.append(coordinate)
        if box_coords[0] == box_coords[2]:
            if box_coords[2] != self.width:
                box_coords[2] += 5
            else:
                box_coords[0] -= 5
        if box_coords[1] == box_coords[3]:
            if box_coords[3] != self.width:
                box_coords[3] += 5
            else:
                box_coords[1] -= 5
        return box_coords
                    
    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name + '.png')
        # read the image
        # print(image_path)
        # image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image = cv2.imread(image_path)
        # plt.imshow(image)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        # plt.imshow(image_resized)
        
        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        image_annotation_dict = self.filenames[image_name]
        
        boxes = []
        if image_annotation_dict["bboxes"]:
            for box_coordinates in self.filenames[image_name]["bboxes"]:
                if box_coordinates:
                    xmin = box_coordinates[0][0]
                    ymin = box_coordinates[0][1]
                    xmax = box_coordinates[1][0]
                    ymax = box_coordinates[1][1]
                    
                    xmin_final = (xmin/image_width)*self.width
                    xmax_final = (xmax/image_width)*self.width
                    ymin_final = (ymin/image_height)*self.height
                    ymax_final = (ymax/image_height)*self.height
                    box_coordinates_loc = [xmin_final, ymin_final, xmax_final, ymax_final]
                    boxes.append(self.check_bounding_boxes(box_coordinates_loc))
        labels_orig = image_annotation_dict["vehicle_class"]
        labels = []
        for label in labels_orig:
            labels.append(label + 1)

        num_objs = len(boxes)
        
        if not boxes:
            # print("Came Inside")
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area = torch.zeros(0, dtype=torch.float32)
        else:
            # bounding box to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # labels to tensor
            labels = torch.as_tensor(labels, dtype=torch.int64)
            # area of the bounding boxes
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # no crowd instances
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        if self.transforms:
            if num_objs != 0:
                try:
                    sample = self.transforms(image = image_resized,
                                             bboxes = boxes,
                                             labels = labels)
                    target_boxes = []
                    for box in sample['bboxes']:
                        target_box = []
                        for box_coordinate in box:
                            if box_coordinate < 1:
                                target_box.append(0)
                            elif box_coordinate > self.width:
                                target_box.append(self.width)
                            else:
                                target_box.append(box_coordinate)
                        if target_box[0] == target_box[2]:
                            if target_box[2] != self.width:
                                target_box[2] += 5
                            else:
                                target_box[0] -= 5
                        if target_box[3] == target_box[1]:
                            if target_box[3] != self.height:
                                target_box[3] += 5
                            else:
                                target_box[1] -= 5
                        target_boxes.append(target_box)
                    # target['boxes'] = torch.Tensor(sample['bboxes'])
                    target['boxes'] = torch.Tensor(target_boxes)
                except:
                    sample = dict()
                    sample['image'] = torch.tensor(image_resized).permute(2,0,1)
                    print("Boxes Exception:", boxes)
            else:
                sample = self.transforms_without_boxes(image = image_resized)
                target['boxes'] = torch.zeros((0,4), dtype=torch.float32)
                target['labels'] = torch.zeros(0, dtype=torch.int64)
                target['area'] = torch.zeros(0, dtype=torch.float32)
            image_resized = sample['image']
        return image_resized, target

    def __len__(self):
        return len(self.all_images)
    
# prepare the final datasets and data loaders
train_dataset = MicrocontrollerDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform(), get_train_transform_without_boxes())
valid_dataset = MicrocontrollerDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform(), get_train_transform_without_boxes())
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")


# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = MicrocontrollerDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")

    # function to visualize a single sample
    def visualize_sample(image, target):
        sampled_image = image.permute(1,2,0)
        boxes = target['boxes']
        labels = target['labels']
        sampled_image = cv2.cvtColor(np.float32(sampled_image), cv2.COLOR_RGB2GRAY)
        for box, label in zip(boxes, labels):
            cv2.rectangle(sampled_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (0, 255, 0), 1)
        # cv2.putText(
        #     image, label, (int(box[0]), int(box[1]-5)), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        # )
        plt.imshow(sampled_image)
        plt.imshow(sampled_image)
        cv2.waitKey(0)

    # NUM_SAMPLES_TO_VISUALIZE = 5
    # for i in range(NUM_SAMPLES_TO_VISUALIZE):
    #     image, target = train_dataset[i]
    #     visualize_sample(image, target)

    image, target = train_dataset[10]
    visualize_sample(image, target)
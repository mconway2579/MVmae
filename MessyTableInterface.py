import json
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import numpy as np

class MessyTableDataset(Dataset):
    def __init__(self, file_path, set_size=2, min_pixel_area=2000, img_shape = (224, 224)):
        self.image_shape = img_shape
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: Image.fromarray(img)),
            transforms.Resize(self.image_shape),
            transforms.ToTensor()
        ])
        self.min_pixel_area = min_pixel_area
        self.set_size = set_size
        #open data file
        data = None
        with open(file_path, 'r') as file:
            data = json.load(file)

        #load intrinsics
        self.intrinics = data['intrinsics']
        for key in self.intrinics:
            self.intrinics[key] = np.array(self.intrinics[key]).reshape(3,3)

        #Check if file already exists
        split = file_path.split('/')[-1].split('.')[0]
        csv_path = f'./MessyTableData/csvs/{split}.csv'
        if os.path.exists(csv_path):
            print(f"loading csv from {csv_path}")
            self.instance_df = pd.read_csv(csv_path)
            return
        
        #If csv file does not exist create it
        print(f"generating csv at {csv_path}")
        column_strings = ['instance',"scene", "label"]
        for i in range(1, 10):
            column_strings += [f"camera_{i}_file", f"camera_{i}_extrinsics", f"camera_{i}_bbox"]
        self.instance_df = pd.DataFrame(columns=column_strings)


        #iterate over the scenes in the splot
        scenes = data['scenes']
        for scene in tqdm(scenes):
            scene_instance_labels = scenes[scene]['instance_summary']
            #create datastructure of {instanceid:{row data}}
            instance_entries = {k:{"instance":k, "label":v, 'scene':scene} for k,v in scene_instance_labels.items()}

            #iterate over the cameras in the scene
            cameras = scenes[scene]['cameras']
            for camera_id in cameras:
                camera = cameras[camera_id]
                camera_path = camera['pathname']
                camera_extrinsics = np.array(camera['extrinsics'])
                for entry in instance_entries:
                    #populate datastructure with {instanceid:{camera_file:camera_path, camera_extrinsics:camera_extrinsics}}
                    instance_entries[entry][f'camera_{camera_id}_file'] = camera_path
                    instance_entries[entry][f'camera_{camera_id}_extrinsics'] = camera_extrinsics
                
                #iterate over the instances in the camera
                for instance_id in camera['instances']:
                    instance = camera['instances'][instance_id]
                    #get the bounding box
                    pos = instance['pos']
        
                    #populate datastructure with {instanceid:{camera_bbox:pos}}
                    instance_entries[instance_id][f'camera_{camera_id}_bbox'] = pos
            
            for entry in instance_entries.values():
                self.instance_df = pd.concat([self.instance_df, pd.DataFrame([entry])], ignore_index=True)
        self.instance_df.to_csv(csv_path, index=False)

        
        # Load and display the image with bounding box
    def debug(self):
        images, bboxes, object_imgs, projection_matricies, instance_scene = self.__getitem__(random.randint(0, len(self.instance_df)-1))
        fig, axes = plt.subplots(nrows=1, ncols=self.set_size, figsize=(20, 20))
        for i, (img, bbox, object_img) in enumerate(zip(images, bboxes, object_imgs)):
            
            #print(f"{bbox=}")

            image = img.permute(1, 2, 0)
            object_img = object_img.permute(1, 2, 0)
            #print(f"{image.shape=}")
            #print(f"{object_img.shape=}")
            new_img = torch.zeros((image.shape[0] + object_img.shape[0] + 1, max(image.shape[1], object_img.shape[1]) + 1, 3))
            new_img[:image.shape[0], :image.shape[1], :] = image
            new_img[image.shape[0]:image.shape[0] + object_img.shape[0], :object_img.shape[1], :] = object_img

            x1, y1, x2, y2 = bbox
            bb_color = torch.tensor([1.0, 0.0, 0.0])
            new_img[y1, x1:x2, :] = bb_color
            new_img[y2, x1:x2, :] = bb_color
            new_img[y1:y2, x1, :] = bb_color
            new_img[y1:y2, x2, :] = bb_color


            axes[i].imshow(new_img)
            axes[i].axis('off')

        fig.suptitle(instance_scene)
        plt.show(block = False)
        plt.pause(1)

    def __len__(self):
        return len(self.instance_df)

    def is_valid_bbox(self, bbox):
        if type(bbox) == str:
            bbox = [float(item.strip()) for item in bbox.strip('[]').split(',')]
        bbox = [int(np.ceil(x)) for x in bbox]
        x1, y1, x2, y2 = bbox
        H = y2 - y1
        W = x2 - x1
        valid = W * H > self.min_pixel_area
        #print(f"{bbox}, {W=}, {H=} {valid}")
        return valid
    def get_valid_cameras(self, row):
        non_nan_cameras = [col.split('_')[1] for col in row.index if col.endswith('_bbox') and np.any(pd.notna(row[col]))]
        valid_cameras = [camera for camera in non_nan_cameras if self.is_valid_bbox(row[f'camera_{camera}_bbox'])]
        return valid_cameras
    def __getitem__(self, idx):
        #print(f"{idx=}")
        row = self.instance_df.iloc[idx]
        #print(f"{row=}")
        valid_cameras = self.get_valid_cameras(row)
        if len(valid_cameras) < self.set_size:
            return self.__getitem__(random.randint(0, len(self.instance_df)-1))
        #print(f"{non_nan_cameras=}")
        cameras_to_use = random.sample(valid_cameras, self.set_size)
        #print(f"{cameras_to_use=}")
        images = []
        bboxes = []
        object_imgs = []
        projection_matricies= []
        instance_scene = [f"instance:{row['instance']}_Scene:{row['scene']}" for camera in cameras_to_use]

        for camera in cameras_to_use:
            
            image_path = os.path.join("./MessyTableData/images", row[f'camera_{camera}_file'])
            image = np.array(Image.open(image_path))

            original_bbox = row[f'camera_{camera}_bbox']
            if type(original_bbox) == str:
                original_bbox = [float(item.strip()) for item in original_bbox.strip('[]').split(',')]
            original_bbox = [int(np.ceil(x)) for x in original_bbox]

            x1, y1, x2, y2 = original_bbox
            cropped_image = image[y1:y2, x1:x2].copy()

            


            transformed_image = self.transform(image.copy())
            images.append(transformed_image)

            original_height, original_width = image.shape[:2]
            scale_x = self.image_shape[1] / original_width
            scale_y = self.image_shape[0] / original_height
            scaled_bbox = [int(original_bbox[0] * scale_x), int(original_bbox[1] * scale_y), int(original_bbox[2] * scale_x), int(original_bbox[3] * scale_y)]
            torch_bbox = torch.tensor(scaled_bbox)
            bboxes.append(torch_bbox)

            transformed_obj_image = self.transform(cropped_image.copy())
            object_imgs.append(transformed_obj_image)


            
            E = row[f'camera_{camera}_extrinsics']
            if type(E) == str:
                E = [float(item.strip()) for item in E.strip('[]').split()]
            x, y, z, roll, pitch, yaw = E
            R_x = np.array([[1, 0, 0],
                            [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll), np.cos(roll)]])
            
            R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])
            
            R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
            
            R = np.dot(R_z, np.dot(R_y, R_x))
            T = np.array([x, y, z]).reshape(3, 1)
            E = np.hstack((R, T))
            #print(f"{E=}")
            I = self.intrinics[camera]
            #print(f"{I=}")
            P = np.dot(I, E)
            projection_matricies.append(P)
        return images, bboxes, object_imgs, projection_matricies, instance_scene
        
def display_batch(data_loader):
    fig, axes = plt.subplots(ncols = data_loader.batch_size, nrows = data_loader.dataset.set_size)
    data = next(iter(data_loader))
    images_batch, bboxes_batch, object_imgs_batch, projection_matricies_batch, instance_scene_batch = data
    #print(f"{type(images_batch)=}, {type(bboxes_batch)=}, {type(object_imgs_batch)=}, {type(projection_matricies_batch)=}, {type(instance_scene_batch)=}")
    #print(f"{len(images_batch)=}, {len(bboxes_batch)=}, {len(object_imgs_batch)=}, {len(projection_matricies_batch)=}, {len(instance_scene_batch)=}")
    #print(f"{bboxes_batch=}")
    #print(f"{bboxes_batch[0]=}")
    #print(f"{bboxes_batch[0][0]=}")

    for i, (images, bboxes, object_imgs, projection_matricies, instance_scene) in enumerate(zip(images_batch, bboxes_batch, object_imgs_batch, projection_matricies_batch, instance_scene_batch)):
        for j, (image, bbox, object_img, P) in enumerate(zip(images, bboxes, object_imgs, projection_matricies)):
            image = image.permute(1, 2, 0)
            object_img = object_img.permute(1, 2, 0)

            new_img = torch.zeros((image.shape[1] + object_img.shape[1] + 1, max(image.shape[1], object_img.shape[1]) + 1, 3))
            new_img[:image.shape[0], :image.shape[1], :] = image
            new_img[image.shape[0]:image.shape[0] + object_img.shape[0], :object_img.shape[1], :] = object_img

            x1, y1, x2, y2 = bbox
            bb_color = torch.tensor([1.0, 0.0, 0.0])
            new_img[y1, x1:x2, :] = bb_color
            new_img[y2, x1:x2, :] = bb_color
            new_img[y1:y2, x1, :] = bb_color
            new_img[y1:y2, x2, :] = bb_color

            axes[i][j].imshow(new_img)
            axes[i][j].axis('off')
            #print(f"{i=}, {j=}")
    plt.show(block = False)
    plt.pause(1)



if __name__ == "__main__":
    bs = 2
    label_dir = './MessyTableData/labels'
    os.makedirs('./MessyTableData/csvs', exist_ok=True)
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.json'):
            file_path = os.path.join(label_dir, label_file)
            dataset = MessyTableDataset(file_path, set_size=5)
            #dataset.debug()
            data_loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=8)
            for _ in (tqdm(data_loader)):
                pass
            display_batch(data_loader)
            #break
    #plt.show()

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
    def __init__(self, file_path, set_size=2, min_pixel_area=2000, train=False):
        self.image_shapes = [128, 256, 512, 1024, 2048]
        self.max_image_shape = (max(self.image_shapes), max(self.image_shapes))
        if train:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: Image.fromarray(img)),
                transforms.Lambda(lambda img: transforms.Resize(self.get_img_size())(img)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: Image.fromarray(img)),
                transforms.Resize(self.max_image_shape),
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
        else:
            print(f"csv {csv_path} does not exist")
        
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
    def get_img_size(self):
        sidelength = random.choice(self.image_shapes)
        #print(f"{sidelength=}")
        return (sidelength, sidelength)
        
        # Load and display the image with bounding box
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
        
        object_imgs = []
        
        instance_scene = [f"instance:{row['instance']}_Scene:{row['scene']}" for camera in cameras_to_use]

        for camera in cameras_to_use:
            
            image_path = os.path.join("./MessyTableData/images", row[f'camera_{camera}_file'])
            image = np.array(Image.open(image_path))

            original_bbox = row[f'camera_{camera}_bbox']
            if type(original_bbox) == str:
                original_bbox = [float(item.strip()) for item in original_bbox.strip('[]').split(',')]
            original_bbox = [int(np.ceil(x)) for x in original_bbox]

            x1, y1, x2, y2 = original_bbox
            cropped_image = image[y1:y2, x1:x2]

            transformed_obj_image = self.transform(cropped_image)
            object_imgs.append(transformed_obj_image)

        return object_imgs, instance_scene

def custom_collate(batch):
    # print(f"{len(batch)=}")
    # print(f"{len(batch[0])=}")
    # print(f"{len(batch[0][0])=}")
    # print(f"{batch[0][0][0].shape=}")
    # print(f"{batch[0][0][1].shape=}")
    # print(f"{batch[0][0][2].shape=}")

    # print(f"{batch[0][1]=}")
    # print()


    object_imgs = [item[0] for item in batch]
    instance_scene = [item[1] for item in batch]
    return object_imgs, instance_scene


def display_batch(data_loader):
    fig, axes = plt.subplots(ncols = data_loader.batch_size, nrows = data_loader.dataset.set_size)
    object_imgs_batch, instance_scene_batch = next(iter(data_loader))
    print(f"{len(object_imgs_batch)=}, {len(instance_scene_batch)=}")

    for i, (object_imgs_set, instance_scene_set) in enumerate(zip(object_imgs_batch, instance_scene_batch)):
        print(f"{len(object_imgs_set)=}, {len(instance_scene_set)=}")

        for j, (object_img, instance_scene) in enumerate(zip(object_imgs_set, instance_scene_set)):
            #print(f"{object_img.shape=}")
            axes[j,i].imshow(object_img.permute(1,2,0))
            axes[j,i].set_title(f"{instance_scene}\n {object_img.shape}", fontsize=5)
            #print(dir(axes[i,j]))
            axes[j,i].axis('off')
    
    plt.tight_layout()
    plt.show(block = False)
    plt.pause(1)



if __name__ == "__main__":
    bs = 4
    set_size = 3
    label_dir = './MessyTableData/labels'
    os.makedirs('./MessyTableData/csvs', exist_ok=True)
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.json'):
            file_path = os.path.join(label_dir, label_file)
            dataset = MessyTableDataset(file_path, set_size=set_size, train=True)
            #dataset.debug()
            data_loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0, collate_fn=custom_collate)
            #for _ in (tqdm(data_loader)):
            #    pass
            display_batch(data_loader)
            #break
    #plt.show()

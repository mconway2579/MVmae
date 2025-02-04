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
    def __init__(self, file_path, set_size=2):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

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
        csv_path = f'./MessyTable/csvs/{split}.csv'
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
        random_row = self.instance_df.sample(n=1).iloc[0]
        non_nan_cameras = self.get_non_nan_cameras(random_row)
        while len(non_nan_cameras) < self.set_size:
            random_row = self.instance_df.sample(n=1).iloc[0]
            non_nan_cameras = self.get_non_nan_cameras(random_row)
        cameras_to_use = random.sample(non_nan_cameras, self.set_size)
        fig, axes = plt.subplots(nrows=1, ncols=self.set_size, figsize=(20, 20))
        for i, camera_id in enumerate(cameras_to_use):
            camera_path = random_row[f'camera_{camera_id}_file']
            bbox = random_row[f'camera_{camera_id}_bbox']
            if type(bbox) == str:
                bbox = [float(item.strip()) for item in bbox.strip('[]').split(',')]
                bbox = [int(np.ceil(x)) for x in bbox]
            print(f"{bbox=}")
            image = Image.open(os.path.join("./MessyTable/images", camera_path))
            axes[i].imshow(image)
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
        plt.show()
        print(f"displayed in debug")

    def __len__(self):
        return len(self.instance_df)


    def get_non_nan_cameras(self, row):
        non_nan_cameras = [col.split('_')[1] for col in row.index if col.endswith('_bbox') and np.any(pd.notna(row[col]))]
        return non_nan_cameras
    def __getitem__(self, idx):
        #print(f"{idx=}")
        row = self.instance_df.iloc[idx]
        #print(f"{row=}")
        non_nan_cameras = self.get_non_nan_cameras(row)
        if len(non_nan_cameras) < self.set_size:
            return self.__getitem__(random.randint(0, len(self.instance_df)-1))
        #print(f"{non_nan_cameras=}")
        cameras_to_use = random.sample(non_nan_cameras, self.set_size)
        #print(f"{cameras_to_use=}")
        images = []
        bboxes = []
        for camera in cameras_to_use:
            image_path = os.path.join("./MessyTable/images", row[f'camera_{camera}_file'])
            #print(f"{image_path=}")
            image = Image.open(image_path)
            images.append(self.transform(image))
            bbox = row[f'camera_{camera}_bbox']
            bbox = [float(item.strip()) for item in bbox.strip('[]').split(',')]
            bbox = [int(np.ceil(x)) for x in bbox]
            bboxes.append(bbox)
            #x1, y1, x2, y2 = bbox
            #print(f"{bbox=}")

            #cropped_image = image[x1:x2, y1:y2]
            #transformed_image = self.transform(cropped_image)
            #images.append(transformed_image)
        return images, bboxes
        

if __name__ == "__main__":
    bs = 4
    label_dir = './MessyTable/labels'
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.json'):
            file_path = os.path.join(label_dir, label_file)
            dataset = MessyTableDataset(file_path, set_size=4)
            dataset.debug()
    """
            dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)
            fig, axes = plt.subplots(nrows=bs, ncols=dataset.set_size, figsize=(20, 20))


            for i, batch in enumerate(dataloader):
                print(f"{len(batch)=}")
                print(f"{len(batch[0])=}")
                print(f"{len(batch[1])=}")

                #print(f"{batch=}")
                for images, bboxes in zip(batch[0], batch[1]):
                    for j, (image, bbox) in enumerate(zip(images, bboxes)):
                        print(f"{image.shape=}")
                        print(f"{bbox.shape=}")
                        axes[j, i].imshow(image.permute(1, 2, 0))
                        x1, y1, x2, y2 = bbox
                        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                        axes[j, i].add_patch(rect)

                plt.show()
                break
            break
        break
    """
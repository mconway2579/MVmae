import json
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MessyTableDataset(Dataset):
    def __init__(self, file_path):
        data = None
        with open(file_path, 'r') as file:
            data = json.load(file)

        self.intrinics = data['intrinsics']
        for key in self.intrinics:
            self.intrinics[key] = np.array(self.intrinics[key]).reshape(3,3)
        #for key in self.intrinics:
        #    print(f"{key}:\n{self.intrinics[key]}")
        #    print()
        scenes = data['scenes']
        self.instance_df = pd.DataFrame(columns=['instance',"scene", "label",
                                                 "camera_1_file", "camera_1_extrinsics", "camera_1_bbox",
                                                ])
        for scene in scenes:
            scene_instance_labels = scenes[scene]['instance_summary']
            #print(f"{scene_instance_labels=}")
            instance_entries = {k:{"label":v, 'scene':scene} for k,v in scene_instance_labels.items()}

            cameras = scenes[scene]['cameras']
            for camera_id in cameras:
                #print(f"{camera_id=}")
                camera = cameras[camera_id]
                camera_path = camera['pathname']
                camera_extrinsics = np.array(camera['extrinsics'])
                #print(f"{camera['instances'].items()=}")
                for entry in instance_entries:
                    instance_entries[entry]['camera_1_file'] = camera_path
                    instance_entries[entry]['camera_1_extrinsics'] = camera_extrinsics
                print(f"{len(camera['instances'])=}")
                print(f"{camera['instances'].keys()=}")
                print(f"{scenes[scene]['instance_summary'].keys()=}")

                max_cam_instance = int(max(camera['instances'].keys(), key=int))
                max_scene_instance = int(max(scenes[scene]['instance_summary'].keys(), key=int))
                assert max_cam_instance <= max_scene_instance, f"{max_cam_instance=} !<={max_scene_instance=}"
                min_cam_instance = int(min(camera['instances'].keys(), key=int))
                min_scene_instance = int(min(scenes[scene]['instance_summary'].keys(), key=int))
                assert min_cam_instance >= min_scene_instance, f"{min_cam_instance=} !<={min_scene_instance=}"
                
                print(f"{len(camera['instances'])=}")
                print(f"{camera['instances'].keys()=}")

                print(f"{len(camera['corners'])=}")
                print(f"{camera['corners'][0]=}")


                print("\n\n")
                img = cv2.imread(os.path.join('./MessyTable/images', camera_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)

                for instance_id in camera['instances']:
                    instance = camera['instances'][instance_id]
                    pos = instance['pos']
                    #print(instance)
                    #print(f"{x1=}, {x2=}, {x3=}, {x4=}")
                    x1, y1, x2, y2 = pos
                    vert_spacing = 50
                    alt_color = 0.3
                    plt.text(x1, y1+vert_spacing, f"{instance['cls']}", color=(1, alt_color, alt_color), fontsize=6)
                    plt.text(x1, y1+(2*vert_spacing), f"{instance['subcls']}", color=(alt_color, alt_color, 1), fontsize=6)
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)

                plt.show()

                #for corner in camera['corners']:
                #    print(f"{corner=}")
                    #instance_entries[instance["subcls"]]['camera_1_bbox']
            """
            camera_instances = camera['instances']
            #camera_corners = camera['corners']
            print(f"{camera_path=}")
            print(f"{camera_extrinsics=}")
            print(f"{camera_instances=}")
            #print(f"{camera_corners=}")
            """

        #print(f"{data['scenes']['20191218-01084-01']["instance_summary"]=}")
        #print(f"{data['scenes']['20191218-01084-01']["cameras"]['1']['pathname'].keys()=}")

        #print(json.dumps(data, indent=4, sort_keys=True))

if __name__ == "__main__":
    train_path = './MessyTable/labels/test_hard.json'
    train_data = MessyTableDataset(train_path)

    #val_path = './MessyTable/labels/val.json'
    #val_data = load_data(val_path)

    #test_path = './MessyTable/labels/test.json'
    #test_data = load_data(test_path)
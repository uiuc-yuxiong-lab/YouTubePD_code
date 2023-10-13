import requests
import os
import pickle
from tqdm import tqdm 
import pathlib
import torchvision
from PIL import Image
root="../PD_raw_data/"
api_key = 'your api key'
api_secret = 'your api secret'
import time
def main():
    mode = "train"
    csv_path =  os.path.join(root,f"{mode}.csv")
    write_path = os.path.join(root,f"{mode}_set/imgs")
    pathlib.Path(write_path).mkdir(parents=True, exist_ok=True)
    with open(csv_path) as file:
        sets = file.readlines()
    sets = [s.split(" ") for s in sets]
    data_path = [os.path.join(root,s[0].split("/")[-1]) for s in sets]
    labels = [int(s[1]) for s in sets]
    video = torchvision.io.VideoReader(data_path[0], "video")
    frames = [] 
    for vid, video_path in  enumerate(data_path):
        video_name = video_path.split("/")[-1].split(".")[0]
        try:
            video = torchvision.io.VideoReader(video_path, "video")

            video.set_current_stream("video")
            frames = [] 
            for frame in video:
                frames.append(frame["data"])
            if len(frames) <= 10:
                frames = frames
            else:
                frames = frames[::len(frames)//10][:10]
        except:
            print(f"{video_name} can't be fount!")
            continue
        # print(type(frames[0]))
        for idx, frame in  enumerate(frames):
            # print(frame)
            im = Image.fromarray(frame.permute(1,2,0).numpy())
            im.save(os.path.join(write_path,f"{video_name}_frame{idx}_class{labels[vid]}.png"))
            
for set_name in ["train_set","val_set","test_set"]:
    # print(os.listdir(os.path.join(root,set_name)))
    file_path = os.path.join(root,set_name)
    img_path = os.path.join(file_path,"imgs")
    kpts_path = os.path.join(file_path,"kpts2")
    os.makedirs(kpts_path,exist_ok=True)
    print(os.listdir(img_path))
    for file in tqdm(os.listdir(img_path)):
        if file.split('.')[1] == 'png':
            pic_path = os.path.join(img_path,file) 
            para = {
                'api_key': (None, api_key),
                'api_secret': (None, api_secret),
                'image_file': open(pic_path, 'rb'),
                'return_landmark': (None, '2'),
            }
            response = requests.post('https://api-us.faceplusplus.com/facepp/v3/detect', files=para).json()
            name = file.split('.')[0] 
            # print(name)
            # print(pic_path)
            # print(response)
            if(len(response['faces'])!=0):
                with open(f'{kpts_path}/{name}.pickle', 'wb') as handle:
                    pickle.dump(response['faces'][0]['landmark'], handle)
            time.sleep(0.1)
import numpy as np        
for set_name in ["train_set","val_set","test_set"]:
    # print(os.listdir(os.path.join(root,set_name)))
    file_path = os.path.join(root,set_name)
    kpts_path = os.path.join(file_path,"kpts2")
    kpts_abs_path = os.path.join(file_path,"kpts_abs")
    os.makedirs(kpts_abs_path,exist_ok=True)
    # print(os.listdir(img_path))
    for file in tqdm(os.listdir(kpts_path)):
        if file.split('.')[1] == 'pickle':
            pickle_p = os.path.join(kpts_path,file) 
            with open(pickle_p,"rb") as f:
                lands = pickle.load(f)

            xs = np.array([l["x"] for l in lands.values()])-lands["nose_middle_contour"]['x']
            ys = np.array([l["y"] for l in lands.values()])-lands["nose_middle_contour"]['y']

            abs_land = np.stack((xs,ys),axis=1)
            # print(abs_land.shape)
            # exit()
            name = file.split('.')[0] 

            np.save(f"{kpts_abs_path}/{name}.npy",abs_land)

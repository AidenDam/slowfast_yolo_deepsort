# python yolo2ava2.py --yolo_path /root/5k_HRW_yolo_Dataset --ava_path /root/autodl-tmp/SCB-ava-Dataset4
import os
import csv
import shutil

import cv2

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--yolo_path', default='./yolo_dataset', type=str)
parser.add_argument('--ava_path', default='./ava_dataset', type=str)
parser.add_argument('--video_path', required=True, help='video path to extract video', type=str)

arg = parser.parse_args()

yolo_path = arg.yolo_path
ava_path = arg.ava_path
video_path = arg.video_path

# SCB_train_predicted_boxes.csv
# SCB_val_predicted_boxes.csv
# SCB_train.csv
# SCB_val.csv
# SCB_train_excluded_timestamps.csv
# SCB_val_excluded_timestamps.csv
# SCB_action_list.pbtxt
# SCB_included_timestamps.txt
# SCB_frame_train.csv
# SCB_frame_val.csv

# https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_train_predicted_boxes.csv


'''
SCB_train_predicted_boxes.csv
SCB_val_predicted_boxes.csv

-5KQ66BBWC4,0902,0.326,0.185,0.470,0.887,80,0.996382
-5KQ66BBWC4,0902,0.326,0.185,0.470,0.887,9,0.996382
-5KQ66BBWC4,0902,0.626,0.153,0.797,0.838,9,0.987177
-5KQ66BBWC4,0902,0.508,0.117,0.648,0.777,9,0.903317
-5KQ66BBWC4,0902,0.222,0.031,0.362,0.529,80,0.983264
...
-5KQ66BBWC4,0916,0.313,0.219,0.666,0.983,79,0.994081
-5KQ66BBWC4,0916,0.086,0.247,0.296,0.989,80,0.989862
-5KQ66BBWC4,0916,0.086,0.247,0.296,0.989,74,0.989862
-5KQ66BBWC4,0916,0.086,0.247,0.296,0.989,12,0.989862
...
-5KQ66BBWC4	1798	0	0.28	0.092	0.983	12	0.902175
-5KQ66BBWC4	1798	0.455	0.244	0.604	0.962	-1	0.49741
-FaXLcSFjUI	902	0.059	0.06	0.896	0.952	11	0.997681
-FaXLcSFjUI	902	0.059	0.06	0.896	0.952	79	0.997681
...
...

'''

########################################################

'''
SCB_train.csv
SCB_val.csv
-5KQ66BBWC4	902	0.077	0.151	0.283	0.811	80	1
-5KQ66BBWC4	902	0.077	0.151	0.283	0.811	9	1
-5KQ66BBWC4	902	0.226	0.032	0.366	0.497	12	0
...
-5KQ66BBWC4	923	0.622	0.133	0.807	0.985	9	31
-5KQ66BBWC4	925	0.285	0.233	0.541	0.93	17	32
...

It should be noted here that the second to last column represents the behavior, 
starting from 1, and the first column from the last represents the person ID, starting from 0

'''

########################################################


'''
SCB_action_list.pbtxt

item {
  name: "bend/bow (at the waist)"
  id: 1
}
item {
  name: "crouch/kneel"
  id: 3
}
item {
  name: "dance"
  id: 4
}
item {
  name: "fall down"
  id: 5
}
item {
  name: "get up"
  id: 6
}
'''

####################################

'''
SCB_frame_train.csv
SCB_frame_val.csv


original_vido_id video_id frame_id path labels
-5KQ66BBWC4 0 0 -5KQ66BBWC4/-5KQ66BBWC4_000001.jpg ""
-5KQ66BBWC4 0 1 -5KQ66BBWC4/-5KQ66BBWC4_000002.jpg ""
-5KQ66BBWC4 0 2 -5KQ66BBWC4/-5KQ66BBWC4_000003.jpg ""
...
-5KQ66BBWC4 0 489 -5KQ66BBWC4/-5KQ66BBWC4_000490.jpg ""
-5KQ66BBWC4 0 490 -5KQ66BBWC4/-5KQ66BBWC4_000491.jpg ""
...
1j20qq1JyX4 235 27026 1j20qq1JyX4/1j20qq1JyX4_027027.jpg ""
1j20qq1JyX4 235 27027 1j20qq1JyX4/1j20qq1JyX4_027028.jpg ""
1j20qq1JyX4 235 27028 1j20qq1JyX4/1j20qq1JyX4_027029.jpg ""
1j20qq1JyX4 235 27029 1j20qq1JyX4/1j20qq1JyX4_027030.jpg ""
1j20qq1JyX4 235 27030 1j20qq1JyX4/1j20qq1JyX4_027031.jpg ""
2PpxiG0WU18 236 0 2PpxiG0WU18/2PpxiG0WU18_000001.jpg ""
2PpxiG0WU18 236 1 2PpxiG0WU18/2PpxiG0WU18_000002.jpg ""
2PpxiG0WU18 236 2 2PpxiG0WU18/2PpxiG0WU18_000003.jpg ""
2PpxiG0WU18 236 3 2PpxiG0WU18/2PpxiG0WU18_000004.jpg ""


'''


#################################

'''
ava
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
|_ frame_lists
|  |_ train.csv
|  |_ val.csv
|_ annotations
   |_ [official AVA annotation files]
   |_ ava_train_predicted_boxes.csv
   |_ ava_val_predicted_boxes.csv

'''



'''
Expand pictures through soft connections, because one picture needs to be copied into 120 pictures, 
which consumes a lot of space.
Why is it 120 pictures:
     0, 1, 2, 3, 4
The picture with annotated information is placed in the middle (that is, 2), 
and there is no annotated information in the first 2 seconds and the last 2 seconds.
So there is a length of 4 seconds, then 30 frames are drawn in 1 second, which is 4*30=120 frames.
'''

# Convert the candidate box in yolo format to the upper left corner (x1, y1) 
# and the coordinate value of the lower right corner (x2, y2)
def yolo_to_xyxy(bbox):
    x, y, w, h = bbox
    x1 = (x - w / 2)
    # Ensure that x1 and y1 are not less than 0
    x1 = x1 if x1 > 0 else 0
    y1 = (y - h / 2)
    y1 = y1 if y1 > 0 else 0
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2

def get_cap(file_path, frame_num):
    if not os.path.exists(file_path):
        return None

    cap = cv2.VideoCapture(file_path)

    # get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # check for valid frame number
    if frame_num >= 0 & frame_num <= totalFrames:
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        miss_frame = 0
    else:
        miss_frame = abs(frame_num) + 1

    return cap, miss_frame


if __name__ == "__main__":
    
    
    train_boxes_path = os.path.join(ava_path, 'annotations/person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv')
    val_boxes_path = os.path.join(ava_path, 'annotations/person_box_67091280_iou90/ava_detection_val_boxes_and_labels.csv')
    
    train_path = os.path.join(ava_path, 'annotations/ava_train_v2.2.csv')
    val_path = os.path.join(ava_path, 'annotations/ava_val_v2.2.csv')
    
    train_excluded_path = os.path.join(ava_path, 'annotations/ava_train_excluded_timestamps_v2.2.csv')
    val_excluded_path = os.path.join(ava_path, 'annotations/ava_val_excluded_timestamps_v2.2.csv')
    
    SCB_frame_train_path = os.path.join(ava_path, 'frame_lists/train.csv')
    SCB_frame_val_path = os.path.join(ava_path, 'frame_lists/val.csv')
    
    try:
        os.makedirs(os.path.join(ava_path,'annotations/person_box_67091280_iou90'))
        os.makedirs(os.path.join(ava_path,'frame_lists'))
        os.makedirs(os.path.join(ava_path,'frames'))
        
        
    except:
        pass
    
    
    if not os.path.exists(train_boxes_path):
        train_boxes = open(train_boxes_path,'w',encoding='utf-8')
        val_boxes = open(val_boxes_path,'w',encoding='utf-8')
        
        train = open(train_path,'w',encoding='utf-8')
        val = open(val_path,'w',encoding='utf-8')
        
        train_excluded = open(train_excluded_path,'w',encoding='utf-8')
        val_excluded = open(val_excluded_path,'w',encoding='utf-8')
        
        SCB_frame_train = open(SCB_frame_train_path,'w',encoding='utf-8')
        SCB_frame_val = open(SCB_frame_val_path,'w',encoding='utf-8')
        
        
    else:
        os.remove(train_boxes_path)
        os.remove(val_boxes_path)
        
        os.remove(train_path)
        os.remove(val_path)
        
        os.remove(train_excluded_path)
        os.remove(val_excluded_path)
        
        os.remove(SCB_frame_train_path)
        os.remove(SCB_frame_val_path)
        
        train_boxes = open(train_boxes_path,'w',encoding='utf-8')
        val_boxes = open(val_boxes_path,'w',encoding='utf-8')
        
        train = open(train_path,'w',encoding='utf-8')
        val = open(val_path,'w',encoding='utf-8')
        
        train_excluded = open(train_excluded_path,'w',encoding='utf-8')
        val_excluded = open(val_excluded_path,'w',encoding='utf-8')
        
        SCB_frame_train = open(SCB_frame_train_path,'w',encoding='utf-8')
        SCB_frame_val = open(SCB_frame_val_path,'w',encoding='utf-8')
    frames_path = os.path.join(ava_path,"frames")

    try:
        shutil.rmtree(frames_path)
        os.makedirs(frames_path)
    except:
        pass
    
    try:
        os.makedirs(frames_path)
    except:
        pass


    train_boxes_writer = csv.writer(train_boxes)
    val_boxes_writer = csv.writer(val_boxes)
    
    train_writer = csv.writer(train)
    val_writer = csv.writer(val)
    
    SCB_frame_train_writer = csv.writer(SCB_frame_train, delimiter=' ')
    SCB_frame_val_writer = csv.writer(SCB_frame_val, delimiter=' ')
    
    SCB_frame_train_writer.writerow(['original_vido_id', 'video_id', 'frame_id', 'path', 'labels'])
    SCB_frame_val_writer.writerow(['original_vido_id', 'video_id', 'frame_id', 'path', 'labels'])
    
        
    video_id = 0
    for root, dirs, files in os.walk(yolo_path, topdown=False):
        for name in files:
            if '.txt' in name and 'checkpoint' not in name:
                txt_path = os.path.join(root, name)
                key_name = name.split('.')[0]
                img_path = txt_path.replace("labels", "images")
                img_path = img_path.replace("txt", "jpg")
                if not os.path.exists(img_path):
                    continue
                
                print('process:', name)

                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        line_data = line.split(' ')
                        x1, y1, x2, y2 =  yolo_to_xyxy([float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
                        name_id_xyxy_conf = [key_name, '0002' ,x1, y1, x2, y2,'', 1.0]
                        name_id_xyxy_action_pid = [key_name, '0002' ,x1, y1, x2, y2, int(line_data[0])+1, 1]

                        if 'train' in txt_path:
                            train_boxes_writer.writerow(name_id_xyxy_conf)
                            train_writer.writerow(name_id_xyxy_action_pid)
                        if 'val' in txt_path:
                            val_boxes_writer.writerow(name_id_xyxy_conf)
                            val_writer.writerow(name_id_xyxy_action_pid)

                os.makedirs(os.path.join(ava_path,"./frames",key_name))

                # change path video
                cap, miss_frame = get_cap(video_path, int(key_name.split('_')[0]) - 60)
                ret, frame_tmp = cap.read()
                while not ret: ret, frame_tmp = cap.read()
                for i in range(miss_frame):
                    name_id_xyxy_action_pid = [key_name, video_id , i, os.path.join(key_name,key_name+"_"+str(i+1).zfill(6)+".jpg"),  f'“”']
                    path_dest = os.path.join(ava_path, './frames/',key_name,key_name+"_"+str(i+1).zfill(6)+".jpg")

                    cv2.imwrite(path_dest, frame_tmp)

                    if 'train' in txt_path:
                        SCB_frame_train_writer.writerow(name_id_xyxy_action_pid)
                    if 'val' in txt_path:
                        SCB_frame_val_writer.writerow(name_id_xyxy_action_pid)

                for i in range(miss_frame, 4*30+31):
                    name_id_xyxy_action_pid = [key_name, video_id , i, os.path.join(key_name,key_name+"_"+str(i+1).zfill(6)+".jpg"),  f'“”']
                    path_dest = os.path.join(ava_path, './frames/',key_name,key_name+"_"+str(i+1).zfill(6)+".jpg")

                    if not cap:
                        shutil.copyfile(img_path, path_dest)
                    else:
                        ret, frame = cap.read()
                        if ret:
                            cv2.imwrite(path_dest, frame)
                            frame_tmp = frame
                        else:
                            cv2.imwrite(path_dest, frame_tmp)

                    if 'train' in txt_path:
                        SCB_frame_train_writer.writerow(name_id_xyxy_action_pid)
                    if 'val' in txt_path:
                        SCB_frame_val_writer.writerow(name_id_xyxy_action_pid)

                video_id = video_id + 1

    train_boxes.close()
    val_boxes.close()
    
    train.close()
    val.close()
    
    train_excluded.close()
    val_excluded.close()
    
    train_excluded.close()
    val_excluded.close()
    
    SCB_frame_train.close()
    SCB_frame_val.close()

    print('Done!!!')
import os
import csv
import cv2
import torch
import numpy as np
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths

from yolov8 import YOLOv8
from deep_sort import build_tracker
from slowfast import SlowFast, ava_inference_transform

from utils.videocapture import MyVideoCapture
from utils.visulization import Draw

def preprocess(clip, bboxes, input_size: int):
    def to_tensor(img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)

    clip = [to_tensor(img) for img in clip]
    clip = torch.cat(clip, 0).permute(-1, 0, 1, 2)
    # Preprocess clip and bounding boxes for video action recognition.
    inputs, inp_boxes, _ = ava_inference_transform(clip, bboxes, 
                                                   num_frames=cfg.DATA.NUM_FRAMES,
                                                   crop_size=input_size,
                                                   data_mean=cfg.DATA.MEAN,
                                                   data_std=cfg.DATA.STD,
                                                   slow_fast_alpha=cfg.SLOWFAST.ALPHA)
    # Prepend data sample id for each bounding box.
    # For more details refere to the RoIAlign in Detectron2
    inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
    return [it.unsqueeze(0).to(args.device) for it in inputs], inp_boxes.to(args.device)


def main():
    yolov8_detector = YOLOv8(args.yolo_ckpt, conf_thres=cfg.YOLO.CONF, iou_thres=cfg.YOLO.IOU)
    tracker = build_tracker(cfg, args.tracker_ckpt, use_cuda=(args.device != 'cpu'))
    video_detector = SlowFast(cfg).load_checkpoint(args.slowfast_ckpt, map_location=args.device)

    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map(args.ava_action_list)
    Draw.init_colors(ava_labelnames)

    input_size = yolov8_detector.input_width, yolov8_detector.input_height

    cap = MyVideoCapture(args.input)
    video_writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), round(30.0/cfg.DATA.NUM_FRAMES), cap.shape)
    csv_file  = open(args.output_csv,'w',encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['person', 'time', 'behavior', 'position', 'identity'])
    while not cap.end:
        _, _ = cap.read()
        if cap.idx > 90: break

        if len(cap.stack) == cfg.DATA.NUM_FRAMES:
            print(f"processing {cap.idx//30}th second clips")
            clip = cap.get_video_clip()
            
            # object detection
            bboxes, scores, class_ids = yolov8_detector(clip[0])
            # select class person
            mask = class_ids == 0
            bboxes, scores = bboxes[mask], scores[mask]
            
            if bboxes.shape[0] == 0:
                continue

            # tracking
            # [bbox_idx,x1,y1,x2,y2,track_id]
            tracked = tracker.update(bboxes, scores, cv2.cvtColor(clip[0], cv2.COLOR_BGR2RGB))

            # action detection
            inputs, inp_boxes = preprocess(clip, bboxes, max(input_size))
            with torch.no_grad():
                # (num_bboxes, num_classes)
                preds = video_detector(inputs, inp_boxes).cpu().numpy()
            action_scores, action_names = [], []
            for it in preds:
                idx = it > cfg.SLOWFAST.THRESH_ACT
                action_scores.append(it[idx])
                action_names.append([ava_labelnames[i+1] for i in np.where(idx)[0]])

            
            # visualization
            img = Draw.draw_detections(clip[0], bboxes, action_names, action_scores, tracked)
            video_writer.write(img)
            for i, (action_name, bbox) in enumerate(zip(action_names, bboxes)):
                if len(tracked) > 0:
                    identity = tracked[tracked[:,0]==i]
                    identity = '' if len(identity) == 0 else identity[0][-1]
                else:
                    identity = ''
                csv_writer.writerow([f's{i}', cap.idx/30, ';'.join(action_name), ' '.join(bbox.astype('str')), identity])

    print('Done!!!')
    cap.release()
    video_writer.release()
    csv_file.close()
    print('Saved video to:', args.output_video)
    print('Saved csv to:', args.output_csv)

if __name__ == '__main__':
    from utils.parsers import load_config, parse_args

    global args
    global cfg

    args = parse_args()
    cfg = load_config(args, args.slowfast_cfg_file)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_video = os.path.join(args.output_dir, args.output_video)
    args.output_csv = os.path.join(args.output_dir, args.output_csv)

    if args.input.isdigit():
        print("Using local camera.")
        args.input = int(args.input)
        
    # print(cfg)
    main()
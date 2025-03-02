import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader
import cv2
import random
import json

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from utils.util import zero_rank_print
#from torchvision.io import read_image
from PIL import Image
def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[None, ...]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255


class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,depth_folder,motion_folder,
            sample_size=256, sample_stride=4, sample_n_frames=14,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        random.shuffle(self.dataset)    
        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.depth_folder = depth_folder
        self.motion_values_folder=motion_folder
        print("length",len(self.dataset))
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    




    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]
        
    
    def get_batch(self, idx):
        def sort_frames(frame_name):
            return int(frame_name.split('_')[1].split('.')[0])
    

    
        while True:
            video_dict = self.dataset[idx]
            videoid = video_dict['videoid']
    
            preprocessed_dir = os.path.join(self.video_folder, videoid)
            depth_folder = os.path.join(self.depth_folder, videoid)
            motion_values_file = os.path.join(self.motion_values_folder, videoid, videoid + "_average_motion.txt")
    
            if not os.path.exists(depth_folder) or not os.path.exists(motion_values_file):
                idx = random.randint(0, len(self.dataset) - 1)
                continue
    
            # Sort and limit the number of image and depth files to 14
            image_files = sorted(os.listdir(preprocessed_dir), key=sort_frames)[:14]
            depth_files = sorted(os.listdir(depth_folder), key=sort_frames)[:14]
    
            # Check if there are enough frames for both image and depth
            if len(image_files) < 14 or len(depth_files) < 14:
                idx = random.randint(0, len(self.dataset) - 1)
                continue
    
            # Load image frames
            numpy_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, img))) for img in image_files])
            pixel_values = numpy_to_pt(numpy_images)
    
            # Load depth frames
            numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(depth_folder, df))) for df in depth_files])
            depth_pixel_values = numpy_to_pt(numpy_depth_images)
    
            # Load motion values
            with open(motion_values_file, 'r') as file:
                motion_values = float(file.read().strip())
    
            return pixel_values, depth_pixel_values, motion_values

        
        
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        #while True:
           # try:
        pixel_values, depth_pixel_values,motion_values = self.get_batch(idx)
           #     break
          #  except Exception as e:
          #      print(e)
          #      idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values,motion_values=motion_values)
        return sample
    
    
class Trajectory_Data(Dataset):
    def __init__(
        self,
        path: str = "./data",
        fps: int = 8,
        sample_size = (256, 256),
    ):
        self.data_path = path
        self.image_path = os.path.join(self.data_path, "images")
        self.trajectory_path = os.path.join(self.data_path, "traj_vid")
        
        self.vids = sorted(os.listdir(self.data_path))
        sub_vids = []
        for trajectory_folder in self.vids:
            if os.path.exists(os.path.join(self.data_path, trajectory_folder, "traj_vid")):
                sub_vids.append(max(0, len(os.listdir(os.path.join(self.data_path, trajectory_folder, "traj_vid")))-1))  
            else:
                sub_vids.append(0)  
        self.sub_vids = sub_vids
        
        self.fps = fps
        
        self.pixel_transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(sample_size),
                # transforms.CenterCrop(sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        
    def _get_image_video_index(self, img_idx):
        # 2 3 0 0 3 4
        cumulative_frames = 0
        for video_index, num_frames in enumerate(self.sub_vids):
            if img_idx < cumulative_frames + num_frames:
                frame_in_video = img_idx - cumulative_frames
                # print(img_idx, cumulative_frames, num_frames)
                return video_index, frame_in_video
            cumulative_frames += num_frames
        print("error for input img_index !!!!")
        return None, None
    
    def get_metadata(self, idx, frame_len=14):
        
        vid_idx, sub_vid_idx = self._get_image_video_index(idx)
        vid_name = self.vids[vid_idx]
        traj_sub_basename = sorted(os.listdir(os.path.join(self.data_path, vid_name, "traj_vid")))[sub_vid_idx]
        # print(traj_sub_basename, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        parts = traj_sub_basename.split("_")
        if len(parts) >= 2:
            start_idx, end_idx = int(parts[0]), int(parts[1])
            end_idx = start_idx+14
        else:
            start_idx, end_idx = None, None
            print("Error: The string does not contain enough parts.")
            
        all_bboxs = np.load(os.path.join(self.data_path, vid_name, "bbox.npy"))
        bboxs = all_bboxs[start_idx] 
            
        cur_frames = sorted(os.listdir(os.path.join(self.data_path, vid_name, "images")))
        cur_trajs = sorted(os.listdir(os.path.join(self.data_path, vid_name, "traj_vid", traj_sub_basename)))
        frame_seq = []
        img_key_seq = []
        
        for frame_idx in range(start_idx, end_idx):
            frame = pil_image_to_numpy(Image.open(os.path.join(self.data_path, vid_name, "images", cur_frames[frame_idx])))
            frame_seq.append(frame)
            img_key_seq.append(f"{vid_name}_{vid_idx}_{frame_idx}")
            
        frame_seq = np.array(frame_seq)

        trajectory_img_seq = []
        for frame_idx in range(end_idx-start_idx-1):
            trajectory_img = pil_image_to_numpy(Image.open(os.path.join(self.data_path, vid_name, "traj_vid", traj_sub_basename, cur_trajs[frame_idx])))
            trajectory_img_seq.append(trajectory_img)
            
        # width, height = trajectory_img_seq[0].width, trajectory_img_seq[0].height
        padding_black = np.zeros(trajectory_img_seq[0].shape)
        trajectory_img_seq.append(padding_black)
        trajectory_img_seq = np.array(trajectory_img_seq)

        # bbox_img_seq = []
        # for frame_idx in range(end_idx-start_idx):
        #     bbox_img = pil_image_to_numpy(Image.open(os.path.join(self.data_path, vid_name, "bbox", traj_sub_basename, cur_trajs[frame_idx])))
        #     trajectory_img_seq.append(bbox_img)
            
        # bbox_img_seq = np.array(bbox_img_seq)
        
        assert len(frame_seq) == len(trajectory_img_seq)
        meta_data = {}
        meta_data["img_seq"] = frame_seq
        meta_data["img_key"] = img_key_seq[0]
        meta_data["trajectory_img_seq"] = trajectory_img_seq
        # meta_data["tbbox_img_seq"] = bbox_img_seq

        
        return meta_data

    @staticmethod
    def __getname__(): return 'traj_folder'

    def __len__(self):
        return sum(self.sub_vids)
        # return 4
    def __getitem__(self, index):
        raw_data = self.get_metadata(index)
        video = raw_data["img_seq"]
        img_key = raw_data["img_key"]
        trajectories = raw_data["trajectory_img_seq"]
        motion_values = 128
        
        pixel_values = numpy_to_pt(video)
        pixel_values = self.pixel_transforms(pixel_values)
        trajectories = numpy_to_pt(trajectories)
        trajectories = self.pixel_transforms(trajectories)
        
        sample = dict(pixel_values=pixel_values, trajectories=trajectories,motion_values=motion_values,img_key=img_key)
        return sample
    
class Trajectory_blender_Data(Dataset):
    def __init__(
        self,
        path: str = "./data",
        fps: int = 8,
        sample_size = (320, 576),
        # sample_size = (256, 320),
        repeat_times = 2,
        frame_length = 14,
        return_rot = False,
        return_bbox = False,
        images_bbox = False,
        depth_mode = False,
        depth_bbox = False,
        mask_initial = False,
        filter_num = -1,
        cut_num = -1,
    ):
        self.data_path = path
        if not (images_bbox or depth_mode):
            self.images_folder = "images"
        elif images_bbox:
            self.images_folder = "images_bbox"
        elif depth_mode:
            if depth_bbox:
                self.images_folder = "depth_maps_bbox"
            else:
                self.images_folder = "depth_maps"

        self.ori_images_folder = "images"

        self.image_path = os.path.join(self.data_path, self.images_folder)
        self.trajectory_path = os.path.join(self.data_path, "traj_vid_enhanced")
        self.repeat_times = repeat_times
        self.frame_length = frame_length
        self.return_rot = return_rot
        self.return_bbox = return_bbox
        self.mask_initial = mask_initial

        if filter_num == -1:
            vids = sorted(os.listdir(self.data_path))
        else:
            vids = []
            for file in sorted(os.listdir(self.data_path)):
                if int(file.split("_")[-2]) < filter_num:
                    vids.append(file)
        sub_vids = []
        use_vids = []

        for trajectory_folder in vids:
            if os.path.exists(os.path.join(self.data_path, trajectory_folder, "traj_vid_enhanced")):
                if len(os.listdir(os.path.join(self.data_path, trajectory_folder, "traj_vid_enhanced")))-1 < self.frame_length:
                    continue
                else:
                    sub_vids.append(len(os.listdir(os.path.join(self.data_path, trajectory_folder, "traj_vid_enhanced")))-1)  
                    use_vids.append(trajectory_folder)

        self.vids = use_vids
            
        self.sub_vids = sub_vids

        if cut_num > 0:
            print("Warning: cutting dataset number, please check if is ablation exp !!!!!")
            self.vids = self.vids[:cut_num]
            self.sub_vids = self.sub_vids[:cut_num]
        
        self.fps = fps
        
        self.pixel_transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(sample_size),
                # transforms.CenterCrop(sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        
    def _get_image_video_index(self, img_idx):
        video_index = img_idx//self.repeat_times
        frame_in_video = random.randint(0, self.sub_vids[video_index]-self.frame_length)
        
        return video_index, frame_in_video
    
    def get_metadata(self, idx):
        
        vid_idx, sub_vid_idx = self._get_image_video_index(idx)
        vid_name = self.vids[vid_idx]

        if self.return_rot:
            rot_type = vid_name.split("_")[-1]
            if rot_type == "line":
                # not rotation for line trajectory
                rot_id = 0
            else:
                rot_id = 1
        
        end_idx = sub_vid_idx + self.frame_length
        start_idx = sub_vid_idx
            
        cur_frames = sorted(os.listdir(os.path.join(self.data_path, vid_name, self.images_folder)))
        cur_trajs = sorted(os.listdir(os.path.join(self.data_path, vid_name, "traj_vid_enhanced")))
        assert cur_frames[0] == cur_trajs[0]
        frame_seq = []
        img_key_seq = []
        input_images = []
        
        for frame_idx in range(start_idx, end_idx):
            # print(vid_name, "image", cur_frames[frame_idx])
            frame = pil_image_to_numpy(Image.open(os.path.join(self.data_path, vid_name, self.images_folder, cur_frames[frame_idx])))
            frame_seq.append(frame)
            img_key_seq.append(f"{vid_name}_{vid_idx}_{frame_idx}")
            
        frame_seq = np.array(frame_seq)

        input_image = pil_image_to_numpy(Image.open(os.path.join(self.data_path, vid_name, self.ori_images_folder, cur_frames[start_idx])))
        input_images.append(input_image)
        input_images = np.array(input_images)

        trajectory_img_seq = []
        for frame_idx in range(start_idx, end_idx-1):
            # print(vid_name, "traj", cur_trajs[frame_idx])
            trajectory_img = pil_image_to_numpy(Image.open(os.path.join(self.data_path, vid_name, "traj_vid_enhanced", cur_trajs[frame_idx])))
            trajectory_img_seq.append(trajectory_img)

        if self.return_bbox:
            bbox_img = pil_image_to_numpy(Image.open(os.path.join(self.data_path, vid_name, "bbox", cur_trajs[start_idx])))

        if self.mask_initial:
            mask = np.zeros(trajectory_img_seq[0].shape)
            bbox_2d = np.load(os.path.join(self.data_path, vid_name, "bbox_2d.py"))
            x_0, x_1, y_0, y_1 = bbox_2d[0][0], bbox_2d[1][0], bbox_2d[0][1], bbox_2d[1][1]
            mask[x_0:x_1, y_0:y_1] = 1
            initial_mask_image = trajectory_img_seq[0] * mask

            
        # width, height = trajectory_img_seq[0].width, trajectory_img_seq[0].height
        padding_black = np.zeros(trajectory_img_seq[0].shape)
        trajectory_img_seq.append(padding_black)
        trajectory_img_seq = np.array(trajectory_img_seq)
        
        assert len(frame_seq) == len(trajectory_img_seq)
        
        meta_data = {}
        meta_data["img_seq"] = frame_seq
        meta_data["input_images"] = input_images
        meta_data["img_key"] = img_key_seq[0]
        meta_data["trajectory_img_seq"] = trajectory_img_seq
        if self.return_rot:
            meta_data["rot_id"] = rot_id
        if self.return_bbox:
            meta_data["bbox"] = bbox_img
        if self.mask_initial:
            meta_data["initial_mask"] = initial_mask_image
        
        return meta_data

    @staticmethod
    def __getname__(): return 'traj_folder'

    def __len__(self):
        return len(self.vids)*self.repeat_times
        # return 4
    def __getitem__(self, index):
        raw_data = self.get_metadata(index)
        video = raw_data["img_seq"]
        img_key = raw_data["img_key"]
        trajectories = raw_data["trajectory_img_seq"]
        if self.return_rot:
            rot_id = torch.tensor(raw_data["rot_id"])
        if self.return_bbox:
            bbox = raw_data["bbox"]
        motion_values = 128
        
        pixel_values = numpy_to_pt(video)
        pixel_values = self.pixel_transforms(pixel_values)

        input_images = numpy_to_pt(raw_data["input_images"])
        input_images = self.pixel_transforms(input_images)

        trajectories = numpy_to_pt(trajectories)
        trajectories = self.pixel_transforms(trajectories)
        if self.return_bbox:
            bbox = numpy_to_pt(bbox)
            bbox = self.pixel_transforms(bbox)
        if self.mask_initial:
            initial_mask = raw_data["initial_mask"]
            initial_mask = numpy_to_pt(initial_mask)
            initial_mask = self.pixel_transforms(initial_mask)


        
        sample = dict(pixel_values=pixel_values, trajectories=trajectories,motion_values=motion_values,img_key=img_key, input_images=input_images)
        if self.return_rot:
            sample["rot_id"] = rot_id
        if self.return_bbox:
            sample["bbox"] = bbox
        if self.mask_initial:
            sample["initial_mask"] = initial_mask
        return sample

class Trajectory_VIPSeg_Data_old(Dataset):
    def __init__(
        self,
        path: str = "./data",
        fps: int = 8,
        sample_size = (320, 576),
        repeat_times = 4,
        frame_length = 14,
        images_bbox = False,
        mask_initial = False,
        filter_num = -1,
        split_file = "train.txt",
        train_mode = "split_vid",
    ):
        self.data_path = path
        if not images_bbox:
            self.images_folder = "images"
        else:
            self.images_folder = "images_bbox"
        self.image_path = os.path.join(self.data_path, self.images_folder)
        self.trajectory_path = os.path.join(self.data_path, "traj_vid_enhanced")
        self.repeat_times = repeat_times
        self.frame_length = frame_length
        self.mask_initial = mask_initial

        with open(split_file, 'r') as f:
            lines = f.readlines()
        set_videos = []

        for line in lines:
            set_videos.append(line.strip())

        if filter_num == -1:
            vids = sorted(os.listdir(self.data_path))
        else:
            vids = []
            for file in sorted(os.listdir(self.data_path)):
                if int(file.split("_")[-2]) < filter_num:
                    vids.append(file)
        sub_vids = []
        use_vids = []
        sub_splits = []

        for trajectory_folder in vids:
            if os.path.exists(os.path.join(self.data_path, trajectory_folder, "traj_vid")):
                if len(os.listdir(os.path.join(self.data_path, trajectory_folder, "traj_vid"))) <= 0:
                    continue
                else:
                    if trajectory_folder in set_videos:
                        # sub_vids.append(len(os.listdir(os.path.join(self.data_path, trajectory_folder, "traj_enhanced")))-1)  
                        sub_splits.append(len(os.listdir(os.path.join(self.data_path, trajectory_folder, "traj_vid"))))
                        use_vids.append(trajectory_folder)

        self.vids = use_vids
            
        self.sub_vids = sub_vids
        self.sub_splits = sub_splits
        self.train_mode = train_mode
        
        self.fps = fps
        
        self.pixel_transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(sample_size),
                # transforms.CenterCrop(sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        
    def _get_image_video_index(self, img_idx):
        video_index = img_idx//self.repeat_times
        frame_in_video = random.randint(0, self.sub_vids[video_index]-self.frame_length)
        
        return video_index, frame_in_video

    def _get_image_video_index_split(self, img_idx):
        video_index = img_idx//self.repeat_times
        selected_folder = random.randint(0, self.sub_splits[video_index]-1)
        
        return video_index, selected_folder
    
    def get_metadata(self, idx):
        
        vid_idx, sub_vid_idx = self._get_image_video_index(idx)
        vid_name = self.vids[vid_idx]
        
        end_idx = sub_vid_idx + self.frame_length
        start_idx = sub_vid_idx
            
        cur_frames = sorted(os.listdir(os.path.join(self.data_path, vid_name, self.images_folder)))
        cur_trajs = sorted(os.listdir(os.path.join(self.data_path, vid_name, "traj_enhanced")))
        assert cur_frames[0] == cur_trajs[0]
        frame_seq = []
        img_key_seq = []
        
        for frame_idx in range(start_idx, end_idx):
            # print(vid_name, "image", cur_frames[frame_idx])
            frame = pil_image_to_numpy(Image.open(os.path.join(self.data_path, vid_name, self.images_folder, cur_frames[frame_idx])))
            frame_seq.append(frame)
            img_key_seq.append(f"{vid_name}_{vid_idx}_{frame_idx}")
            
        frame_seq = np.array(frame_seq)

        trajectory_img_seq = []
        for frame_idx in range(start_idx, end_idx-1):
            # print(vid_name, "traj", cur_trajs[frame_idx])
            trajectory_img = pil_image_to_numpy(Image.open(os.path.join(self.data_path, vid_name, "traj_enhanced", cur_trajs[frame_idx])))
            trajectory_img_seq.append(trajectory_img)
            
        # width, height = trajectory_img_seq[0].width, trajectory_img_seq[0].height
        padding_black = np.zeros(trajectory_img_seq[0].shape)
        trajectory_img_seq.append(padding_black)
        trajectory_img_seq = np.array(trajectory_img_seq)
        
        assert len(frame_seq) == len(trajectory_img_seq)
        
        meta_data = {}
        meta_data["img_seq"] = frame_seq
        meta_data["img_key"] = img_key_seq[0]
        meta_data["trajectory_img_seq"] = trajectory_img_seq
        if self.mask_initial:
            meta_data["initial_mask"] = initial_mask_image
        
        return meta_data

    def get_metadata_split(self, idx):
        
        vid_idx, sub_vid_idx = self._get_image_video_index_split(idx)
        vid_name = self.vids[vid_idx]
            
        cur_frames = sorted(os.listdir(os.path.join(self.data_path, vid_name, self.images_folder)))
        cur_traj_folder = sorted(os.listdir(os.path.join(self.data_path, vid_name, "traj_vid")))[sub_vid_idx]
        cur_trajs = sorted(os.listdir(os.path.join(self.data_path, vid_name, "traj_vid", cur_traj_folder)))
        start_idx = int(cur_traj_folder.split("_")[0])
        end_idx = start_idx + self.frame_length
        assert cur_frames[0] == cur_trajs[0]
        frame_seq = []
        img_key_seq = []
        
        for frame_idx in range(start_idx, end_idx):
            # print(vid_name, "image", cur_frames[frame_idx])
            frame = pil_image_to_numpy(Image.open(os.path.join(self.data_path, vid_name, self.images_folder, cur_frames[frame_idx])))
            frame_seq.append(frame)
            img_key_seq.append(f"{vid_name}_{vid_idx}_{frame_idx}")
            
        frame_seq = np.array(frame_seq)

        trajectory_img_seq = []
        for frame_idx in range(self.frame_length-1):
            # print(vid_name, "traj", cur_trajs[frame_idx])
            trajectory_img = pil_image_to_numpy(Image.open(os.path.join(self.data_path, vid_name, "traj_vid", cur_traj_folder, cur_trajs[frame_idx])))
            trajectory_img_seq.append(trajectory_img)
            
        # width, height = trajectory_img_seq[0].width, trajectory_img_seq[0].height
        padding_black = np.zeros(trajectory_img_seq[0].shape)
        trajectory_img_seq.append(padding_black)
        trajectory_img_seq = np.array(trajectory_img_seq)
        
        assert len(frame_seq) == len(trajectory_img_seq)
        
        meta_data = {}
        meta_data["img_seq"] = frame_seq
        meta_data["img_key"] = img_key_seq[0]
        meta_data["trajectory_img_seq"] = trajectory_img_seq
        if self.mask_initial:
            meta_data["initial_mask"] = initial_mask_image
        
        return meta_data

    @staticmethod
    def __getname__(): return 'traj_folder'

    def __len__(self):
        return len(self.vids)*self.repeat_times
        # return 4
    def __getitem__(self, index):
        if self.train_mode == "split_vid":
            raw_data = self.get_metadata_split(index)
        elif sefl.train_mode == "whole_vid":
            raw_data = self.get_metadata(index)
        else:
            assert False
        video = raw_data["img_seq"]
        img_key = raw_data["img_key"]
        trajectories = raw_data["trajectory_img_seq"]
        # if self.return_rot:
        #     rot_id = torch.tensor(raw_data["rot_id"])
        # if self.return_bbox:
        #     bbox = raw_data["bbox"]
        motion_values = 128
        
        pixel_values = numpy_to_pt(video)
        pixel_values = self.pixel_transforms(pixel_values)
        trajectories = numpy_to_pt(trajectories)
        trajectories = self.pixel_transforms(trajectories)
        # if self.return_bbox:
        #     bbox = numpy_to_pt(bbox)
        #     bbox = self.pixel_transforms(bbox)
        # if self.mask_initial:
        #     initial_mask = raw_data["initial_mask"]
        #     initial_mask = numpy_to_pt(initial_mask)
        #     initial_mask = self.pixel_transforms(initial_mask)

        sample = dict(pixel_values=pixel_values, trajectories=trajectories,motion_values=motion_values,img_key=img_key)
        # if self.return_rot:
        #     sample["rot_id"] = rot_id
        # if self.return_bbox:
        #     sample["bbox"] = bbox
        if self.mask_initial:
            sample["initial_mask"] = initial_mask
        return sample
    

class Trajectory_VIPSeg_Data(Dataset):
    def __init__(
        self,
        path: str = "./data",
        fps: int = 8,
        # sample_size = (320, 576),
        sample_size = (256, 320),
        repeat_times = 4,
        frame_length = 14,
        images_bbox = False,
        mask_initial = False,
        filter_num = -1,
        split_file = "train.txt",
        train_mode = "split_vid",
        return_cam = False,
        camera_path = None
    ):
        self.data_path = path
        self.images_folder = "imgs"

        self.trajectory_json = os.path.join(self.data_path, "trajectory_CoTracker_all")
        self.camera_path = camera_path
        self.repeat_times = repeat_times
        self.frame_length = frame_length
        self.return_cam = return_cam

        with open(split_file, 'r') as f:
            lines = f.readlines()
        set_videos = []

        for line in lines:
            set_videos.append(line.strip())

        if filter_num == -1:
            vids = sorted(os.listdir(os.path.join(self.data_path, self.images_folder)))
        else:
            vids = []
            for file in sorted(os.listdir(self.data_path)):
                if int(file.split("_")[-2]) < filter_num:
                    vids.append(file)
        sub_vids = []
        use_vids = []

        for vid_folder in vids:
            if vid_folder in set_videos:
                sub_images = os.path.join(self.data_path, "imgs", vid_folder)
                sub_anno = os.path.join(self.trajectory_json, f"{vid_folder}.json")
                with open(sub_anno, 'r') as json_file:
                    anno_traj = json.load(json_file)
                if os.path.exists(sub_images) and os.path.exists(sub_anno):
                    if len(anno_traj[next(iter(anno_traj))]) < self.frame_length:
                        continue
                    else:
                        if vid_folder in set_videos:
                            sub_vids.append(len(anno_traj[next(iter(anno_traj))]))  
                            use_vids.append(vid_folder)

        self.vids = use_vids
            
        self.sub_vids = sub_vids
        self.train_mode = train_mode
        
        self.fps = fps
        self.sample_size = sample_size
        
        self.pixel_transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(sample_size),
                # transforms.CenterCrop(sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        
    def _get_image_video_index(self, img_idx):
        video_index = img_idx//self.repeat_times
        # print(self.sub_vids[video_index], self.vids[video_index])
        frame_in_video = random.randint(0, self.sub_vids[video_index]-self.frame_length)
        
        return video_index, frame_in_video
    
    def draw_traj(self, vid_name, start_idx, end_idx, size, original_size):
        # print(f"target size {size}, ori size: {original_size} !!!!!!")
        traj_json = os.path.join(self.trajectory_json, f"{vid_name}.json")
        with open(traj_json, 'r') as json_file:
            trajectory_json = json.load(json_file)
        
        trajectory_list = []
        
        for index in trajectory_json:
            trajectories = trajectory_json[index]
            trajectories = [[int(i[0]/original_size[1]*size[1]),int(i[1]/original_size[0]*size[0])] for i in trajectories]
            trajectory_list.append(trajectories)
            
        trajectory_img_seq = []
        
        for len_index in range(start_idx, end_idx-1):
            mask_img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            for index in range(len(trajectory_list)):
                trajectory_ = trajectory_list[index]
                cv2.line(mask_img, (trajectory_[len_index][0], trajectory_[len_index][1]),(trajectory_[len_index+1][0],trajectory_[len_index+1][1]),(0,0,255),3)
                cv2.circle(mask_img, (trajectory_[len_index+1][0],trajectory_[len_index+1][1]), 3, (0, 255, 0), -1)
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
                
            trajectory_img_seq.append(mask_img)
            
        return trajectory_img_seq
        
    
    def get_metadata(self, idx):
        
        vid_idx, sub_vid_idx = self._get_image_video_index(idx)
        vid_name = self.vids[vid_idx]
        
        end_idx = sub_vid_idx + self.frame_length
        start_idx = sub_vid_idx
            
        cur_frames = sorted(os.listdir(os.path.join(self.data_path, self.images_folder, vid_name)))

        frame_seq = []
        img_key_seq = []
        
        for frame_idx in range(start_idx, end_idx):
            # print(vid_name, "image", cur_frames[frame_idx])
            frame = pil_image_to_numpy(Image.open(os.path.join(self.data_path, self.images_folder, vid_name, cur_frames[frame_idx])))
            frame_seq.append(frame)
            img_key_seq.append(f"{vid_name}_{vid_idx}_{frame_idx}")
            
        frame_seq = np.array(frame_seq)
        
        trajectory_img_seq = self.draw_traj(vid_name, start_idx, end_idx, self.sample_size, frame_seq[0].shape)
            
        # width, height = trajectory_img_seq[0].width, trajectory_img_seq[0].height
        padding_black = np.zeros(trajectory_img_seq[0].shape)
        trajectory_img_seq.append(padding_black)
        trajectory_img_seq = np.array(trajectory_img_seq)
        
        assert len(frame_seq) == len(trajectory_img_seq)

        if self.return_cam:
            assert self.camera_path != None
            
            camera_para = os.path.join(self.camera_path, vid_name, "camera.npy")
            if os.path.exists(camera_para):
                cam_parameter = np.load(camera_para, allow_pickle=True).item()
                # print(cam_parameter.keys())
                cam_R = cam_parameter["pred_cam_R"]
                cam_R = cam_R.reshape(len(cam_R), -1)
                cam_T = cam_parameter["pred_cam_T"]
                if np.isnan(cam_T).any():
                    cam_T = np.zeros(cam_T.shape)
                # print(cam_T)
                cam_parameter = np.concatenate((cam_R, cam_T), axis=-1)[start_idx:end_idx, :]
            else:
                cam_parameter = np.zeros((self.frame_length, 12))
            # norm camera movement based on first frame
            cam_parameter -= cam_parameter[0]
        else:
            cam_parameter = None
        
        meta_data = {}
        meta_data["img_seq"] = frame_seq
        meta_data["img_key"] = img_key_seq[0]
        meta_data["trajectory_img_seq"] = trajectory_img_seq
        meta_data["cam_parameter"] = cam_parameter
        
        return meta_data

    @staticmethod
    def __getname__(): return 'traj_folder'

    def __len__(self):
        return len(self.vids)*self.repeat_times
        # return 4
    def __getitem__(self, index):
        raw_data = self.get_metadata(index)

        video = raw_data["img_seq"]
        img_key = raw_data["img_key"]
        trajectories = raw_data["trajectory_img_seq"]
        # if self.return_rot:
        #     rot_id = torch.tensor(raw_data["rot_id"])
        # if self.return_bbox:
        #     bbox = raw_data["bbox"]
        motion_values = 128
        
        pixel_values = numpy_to_pt(video)
        pixel_values = self.pixel_transforms(pixel_values)
        trajectories = numpy_to_pt(trajectories)
        trajectories = self.pixel_transforms(trajectories)

        if self.return_cam:
            cam_parameter = raw_data["cam_parameter"]
            cam_parameter = torch.tensor(cam_parameter, dtype=torch.float32)
            sample = dict(pixel_values=pixel_values, trajectories=trajectories,motion_values=motion_values,img_key=img_key, cam_parameter=cam_parameter)
        else:
            sample = dict(pixel_values=pixel_values, trajectories=trajectories,motion_values=motion_values,img_key=img_key)
        
        return sample

    




if __name__ == "__main__":
    from utils.util import save_videos_grid

    dataset = WebVid10M(
        csv_path="/data/webvid/results_2M_train.csv",
        video_folder="/data/webvid/data/videos",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=True,
    )
    import pdb
    pdb.set_trace()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)
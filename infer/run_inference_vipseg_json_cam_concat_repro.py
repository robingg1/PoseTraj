import os
import torch
import datetime
import numpy as np
from PIL import Image
from pipeline.pipeline_stable_video_diffusion_controlnet_cam import StableVideoDiffusionPipelineControlNet
from models.controlnet_sdv_cam_infer import ControlNetSDVModel
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
import cv2
import re 
import json

def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]
    
    pil_frames = pil_frames[0]
    duration_ms = int(1000 / fps)*3

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=duration_ms,
                       loop=0)

def save_gifs_side_by_side(batch_output, validation_control_images,output_folder,name = 'none', target_size=(512 , 512),duration=200):

    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    def create_gif(image_list, gif_path, duration=100):
        pil_images = [validate_and_convert_image(img,target_size=target_size) for img in image_list]
        pil_images = [img for img in pil_images if img is not None]
        if pil_images:
            pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], loop=0, duration=duration)

    # Creating GIFs for each image list
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gif_paths = []
    
#     validation_control_images = validation_control_images*255 validation_images, 
    for idx, image_list in enumerate([validation_control_images, flattened_batch_output]):
        
#         if idx==0:
#             continue

        gif_path = os.path.join(output_folder, f"temp_{idx}_{timestamp}.gif")
        create_gif(image_list, gif_path)
        gif_paths.append(gif_path)

    # Function to combine GIFs side by side
    def combine_gifs_side_by_side(gif_paths, output_path):
        print(gif_paths)
        gifs = [Image.open(gif) for gif in gif_paths]

        # Assuming all gifs have the same frame count and duration
        frames = []
        for frame_idx in range(gifs[0].n_frames):
            combined_frame = None
            
                
            for gif in gifs:
                
                gif.seek(frame_idx)
                if combined_frame is None:
                    combined_frame = gif.copy()
                else:
                    combined_frame = get_concat_h(combined_frame, gif.copy())
            frames.append(combined_frame)
        print(gifs[0].info['duration'])
        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=duration)

    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    # Combine the GIFs into a single file
    combined_gif_path = os.path.join(output_folder, f"combined_frames_{name}_{timestamp}.gif")
    combine_gifs_side_by_side(gif_paths, combined_gif_path)

    # Clean up temporary GIFs
    for gif_path in gif_paths:
        os.remove(gif_path)

    return combined_gif_path

# Define functions
def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image

def create_image_grid(images, rows, cols, target_size=(256, 256)):
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid

def tensor_to_pil(tensor):
    """ Convert a PyTorch tensor to a PIL Image. """
    # Convert tensor to numpy array
    if len(tensor.shape) == 4:  # batch of images
        images = [Image.fromarray(img.numpy().transpose(1, 2, 0)) for img in tensor]
    else:  # single image
        images = Image.fromarray(tensor.numpy().transpose(1, 2, 0))
    return images

def save_combined_frames(batch_output, validation_images, validation_control_images, output_folder):
    # Flatten batch_output to a list of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # Convert tensors in lists to PIL Images
    validation_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_images]
    validation_control_images = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in validation_control_images]
    flattened_batch_output = [tensor_to_pil(img) if torch.is_tensor(img) else img for img in batch_output]

    # Flatten lists if they contain sublists (for tensors converted to multiple images)
    validation_images = [img for sublist in validation_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    validation_control_images = [img for sublist in validation_control_images for img in (sublist if isinstance(sublist, list) else [sublist])]
    flattened_batch_output = [img for sublist in flattened_batch_output for img in (sublist if isinstance(sublist, list) else [sublist])]

    # Combine frames into a list
    combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3
    rows = (num_images + cols - 1) // cols

    # Create and save the grid image
    grid = create_image_grid(combined_frames, rows, cols, target_size=(256, 256))
    if grid is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"combined_frames_{timestamp}.png"
        output_path = os.path.join(output_folder, filename)
        grid.save(output_path)
    else:
        print("Failed to create image grid")

def load_images_from_folder(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        # First, try the pattern 'frame_x_7fps'
        new_pattern_match = re.search(r'frame_(\d+)_7fps', filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))

        # If the new pattern is not found, use the original digit extraction method
        matches = re.findall(r'\d+', filename)
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images

def load_images_from_folder_to_pil(folder, target_size=(512, 512)):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    def frame_number(filename):
        # Try the pattern 'frame_x_7fps'
        new_pattern_match = re.search(r'frame_(\d+)_7fps', filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))

        # If the new pattern is not found, use the original digit extraction method
        matches = re.findall(r'\d+', filename)
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder))

    # Load, resize, and convert images
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image with original channels
            if img is not None:
                # Resize image
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # Convert to uint8 if necessary
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)

                # Ensure all images are in RGB format
                if len(img.shape) == 2:  # Grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image in BGR format
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert the numpy array to a PIL image
                pil_img = Image.fromarray(img)
                images.append(pil_img)

    return images

# Usage example
def convert_list_bgra_to_rgba(image_list):
    """
    Convert a list of PIL Image objects from BGRA to RGBA format.

    Parameters:
    image_list (list of PIL.Image.Image): A list of images in BGRA format.

    Returns:
    list of PIL.Image.Image: The list of images converted to RGBA format.
    """
    rgba_images = []
    for image in image_list:
        if image.mode == 'RGBA' or image.mode == 'BGRA':
            # Split the image into its components
            b, g, r, a = image.split()
            # Re-merge in RGBA order
            converted_image = Image.merge("RGBA", (r, g, b, a))
        else:
            # For non-alpha images, assume they are BGR and convert to RGB
            b, g, r = image.split()
            converted_image = Image.merge("RGB", (r, g, b))

        rgba_images.append(converted_image)

    return rgba_images

def trans_numpy(video_frames):
    res = []
    for frame in video_frames:
        res.append(np.array(frame))
    return res


def export_to_video(video_frames, output_video_path, fps):
    print(video_frames[0])
    video_frames = trans_numpy(video_frames[0])
    print(video_frames[0])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def export_to_images(video_frames, output_video_path, fps, width, height):
    basename = os.path.split(output_video_path)[1]
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)
    print(video_frames[0])
    video_frames = trans_numpy(video_frames[0])
    
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        save_path = os.path.join(output_video_path, f"{basename}_{i:05}.png")
        cv2.imwrite(save_path, img)

# Main script
if __name__ == "__main__":
    args = {
        "pretrained_model_name_or_path": "../stable-video-diffusion-img2vid",
        "validation_image_folder": "./validation_demo/rgb",
        "validation_control_folder": "./validation_demo/depth",
        "validation_image": "./validation_demo/chair.png",
        "output_dir": "./output",
        "height": 320,
        "width": 576,
        # "height": 256,
        # "width": 256,
        # "height": 256,
        # "width": 320,
        # cant be bothered to add the args in myself, just use notepad
    }
    images_bbox = False
    output_basename = f'videos_ft_vipseg_5k_{args["width"]}_{args["height"]}_unmasked_100s_roshan_train_fixed_random'

    # Load validation images and control images

    # Load and set up the pipeline
    controlnet = controlnet = ControlNetSDVModel.from_pretrained("/minimax-3d-rw/tuolaji/fyp/svd_trajectory/model_out_vipseg_ft_5k_100s_cam_unmasked_reproduce_roshan/checkpoint-100000",subfolder="controlnet")
    # controlnet = controlnet = ControlNetSDVModel.from_pretrained("/apdcephfs/private_robinggji/workspace/svd_trajectory/model_out_spatial_VIPSeg_finetune_10k/checkpoint-95000",subfolder="controlnet")
    # controlnet = controlnet = ControlNetSDVModel.from_pretrained("/apdcephfs/private_robinggji/workspace/svd_trajectory/model_out_spatial_rotation_only_3d/checkpoint-30000",subfolder="controlnet")
    # controlnet = controlnet = ControlNetSDVModel.from_pretrained("/apdcephfs/private_robinggji/workspace/svd_trajectory/model_out_spatial_rotation_only_3d_nospa/checkpoint-30000",subfolder="controlnet")
    # controlnet = controlnet = ControlNetSDVModel.from_pretrained("/apdcephfs/private_robinggji/workspace/svd_trajectory/model_out_spatial_rotation_3d_imagesbbox_finetune_bbox_cond/checkpoint-30000",subfolder="controlnet")
    # controlnet = controlnet = ControlNetSDVModel.from_pretrained("/apdcephfs/private_robinggji/workspace/svd_trajectory/model_out_spatial_rotation_only_3d_bbox_cond/checkpoint-30000",subfolder="controlnet")
    # controlnet = controlnet = ControlNetSDVModel.from_pretrained("/apdcephfs/private_robinggji/workspace/svd_trajectory/model_out_spatial_rotation_only_3d_imagebbox_single_traj_finetune/checkpoint-30000",subfolder="controlnet")
    # controlnet = controlnet = ControlNetSDVModel.from_pretrained("/apdcephfs/private_robinggji/workspace/svd_trajectory/model_out_spatial_rotation_3d_imagesbbox_finetune_nospa/checkpoint-30000",subfolder="controlnet")
    
    # controlnet = controlnet = ControlNetSDVModel.from_pretrained("/apdcephfs/private_robinggji/workspace/svd_trajectory/model_out_spatial_rotation_3d_davis_finetune/checkpoint-12000",subfolder="controlnet")

    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args["pretrained_model_name_or_path"],subfolder="unet")
    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(args["pretrained_model_name_or_path"],controlnet=controlnet,unet=unet)
    pipeline.enable_model_cpu_offload()
    # Additional pipeline configurations can be added here
    #pipeline.enable_xformers_memory_efficient_attention()
    # Create output directory if it doesn't exist
    val_save_dir = os.path.join(args["output_dir"], f'validation_images_5k_{args["width"]}_{args["height"]}_unmasked_100s_roshan_train_fixed_random')
    os.makedirs(val_save_dir, exist_ok=True)

    # validation_images = load_images_from_folder_to_pil(args["validation_image_folder"])
    #validation_images = convert_list_bgra_to_rgba(validation_images)
    data_path = "/minimax-3d-rw/tuolaji/fyp/dataset/VIPSeg"
    # data_path = "/apdcephfs/private_robinggji/VIPSeg_traj_v4"
    split_path = "/minimax-3d-rw/tuolaji/fyp/dataset/VIPSeg/val.txt"
    camera_path = "../tram/result/vipseg"

    vid_folders = []

    with open(split_path, 'r') as f:
        lines = f.readlines()
    set_videos = []

    for line in lines:
        set_videos.append(line.strip())


    for vid_folder in os.listdir(os.path.join(data_path, "imgs")):
        if vid_folder in set_videos:
            vid_folders.append(vid_folder)
    
    for vid_folder in sorted(vid_folders):

        images = os.path.join(data_path, "imgs", vid_folder)
        all_images = sorted(os.listdir(images))

        if len(all_images)<14:
            print(f"Too short video clips {vid_folder}")
            continue        

        if len(all_images) == 1:
            print("wild data mode")
            start_frame = 0
            basename = vid_folder+f"_{start_frame}"
            output_video_path = os.path.join(args["output_dir"], "videos_vipseg_val", "pred_videos", f"{basename}.mp4")

            start_image = os.path.join(data_path, images, all_images[start_frame])
            validation_image = Image.open(start_image).convert('RGB')

            validation_control_images = []
            for idx in range(start_frame, start_frame+13):
                cur_image = os.path.join(data_path, traj_vid, all_trajs[idx])
                cur_image = Image.open(cur_image).convert('RGB')
                validation_control_images.append(cur_image)

            validation_control_images = validation_control_images[:13]

            padding_black = np.zeros((320, 576, 3), dtype=np.uint8)
            padding_black = Image.fromarray(padding_black)
            validation_control_images.append(padding_black)

            video_frames = pipeline(validation_image, validation_control_images[:14], decode_chunk_size=8,num_frames=14,motion_bucket_id=10,controlnet_cond_scale=1.0,width=args["width"],height=args["height"]).frames
            export_to_images(video_frames, output_video_path, fps=5, width=576, height=320)
            export_to_gif(video_frames, output_video_path, 30)

        else:
            start_frame = 0
            basename = vid_folder+f"_{start_frame}"
            print(start_frame)
            print(basename)

            output_video_path = os.path.join(args["output_dir"], output_basename, "pred_videos", f"{basename}.mp4")
            output_video_gt_path = os.path.join(args["output_dir"], output_basename, "gt", f"{basename}.mp4")

            start_image = os.path.join(data_path, "imgs", vid_folder, all_images[start_frame])

            validation_image = Image.open(start_image).convert('RGB')
            validation_images = [[]]
            for idx in range(start_frame, start_frame+14):
                cur_image = os.path.join(data_path, "imgs", vid_folder, all_images[idx])
                cur_image = Image.open(cur_image).convert('RGB')
                validation_images[0].append(cur_image)

            original_size = np.array(validation_image).shape
            print(original_size)
            # assert False

            validation_control_images = []
            size = [args["height"], args["width"]]
            # size = [args["height"], int(original_size[1]*args["height"]/original_size[0])]

            # json_path = os.path.join(data_path, "trajectory_CoTracker_all", f"{vid_folder}.json")
            # json_path = os.path.join("/minimax-3d-rw/tuolaji/fyp/dataset/VIPSeg/trajectory_CoTracker_all", f"{vid_folder}.json")
            json_path = os.path.join("/minimax-3d-rw/tuolaji/fyp/DragAnything/data/VIPSeg_Test/test_traject", f"{vid_folder}.json")

            with open(json_path, "r") as json_file:
                trajectory_json = json.load(json_file)
            
            trajectory_list = []
            
            for index in trajectory_json:
                trajectories = trajectory_json[index]
                # trajectories = ann["trajectory"]
                trajectories = [[int(i[0]*(size[1]/original_size[1])),int(i[1]*(size[0]/original_size[0]))] for i in trajectories]
                # trajectories = [[int((original_size[0] - i[0])*(size[0]/original_size[0])), int((original_size[1] - i[1])*(size[1]/original_size[1]))] for i in trajectories]
                # print(trajectories)
                trajectory_list.append(trajectories)
        
            for len_index in range(13):
                mask_img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
                for index in range(len(trajectory_list)):
                    trajectory_ = trajectory_list[index]
                    cv2.line(mask_img, (trajectory_[len_index][0], trajectory_[len_index][1]),(trajectory_[len_index+1][0],trajectory_[len_index+1][1]),(0,0,255),3)
                    cv2.circle(mask_img, (trajectory_[len_index+1][0],trajectory_[len_index+1][1]), 3, (0, 255, 0), -1)

                # pil_mask_img = Image.fromarray(cv2.resize(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB), (tar_size[1], tar_size[0])))
                pil_mask_img = Image.fromarray(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB))
                # pil_mask_img = mask_img
                validation_control_images.append(pil_mask_img)
                # print(pil_mask_img.width, pil_mask_img.height)

            padding_image = Image.fromarray(np.zeros(np.array(pil_mask_img).shape, dtype=np.uint8))
            validation_control_images.append(padding_image)

            # camera_para = os.path.join(camera_path, vid_folder, "camera.npy")
            # if os.path.exists(camera_para):
            #     cam_parameter = np.load(camera_para, allow_pickle=True).item()
            #     # print(cam_parameter.keys())
            #     cam_R = cam_parameter["pred_cam_R"]
            #     cam_R = cam_R.reshape(len(cam_R), -1)
            #     cam_T = cam_parameter["pred_cam_T"]
            #     cam_parameter = np.concatenate((cam_R, cam_T), axis=-1)[:14, :]
            # else:
            #     cam_parameter = np.zeros((frame_length, 12))

            cam_parameter = np.zeros((14, 12))
            # norm camera movement based on first frame
            cam_parameter -= cam_parameter[0]
            # cv2.imwrite("tmp.png", np.array(validation_control_images[0]))
            # assert False

            # if args["width"] == 256:
            #     for i in range(len(validation_images)):
            #         validation_images[0][i] = validation_images[0][i].resize((256, 256), Image.BILINEAR)
            #         validation_control_images[i] = validation_control_images[i].resize((256, 256), Image.BILINEAR)

            video_frames = pipeline(validation_image, validation_control_images[:14], cam_parameter[:14], decode_chunk_size=8,num_frames=14,motion_bucket_id=10,controlnet_cond_scale=1.0,width=args["width"],height=args["height"]).frames
            export_to_images(video_frames, output_video_path, fps=5, width=args["width"], height=args["height"])
            export_to_images(validation_images, output_video_gt_path, fps=5, width=args["width"], height=args["height"])
            try:
                save_gifs_side_by_side(video_frames, validation_control_images[:14], val_save_dir, target_size=(args["width"], args["height"]), duration=110)
            except Exception as e:
                print(e)


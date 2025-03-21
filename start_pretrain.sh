CUDA_VISIBLE_DEVICES=1 accelerate launch train_svd_traj_blender_14.py \
        --pretrained_model_name_or_path="../stable-video-diffusion-img2vid" \
        --output_dir="model_out_objaverse_10k_bbox" \
        --video_folder="../render_objaverse_10k" \
        --validation_image_folder="../dataset/objaverse_val" \
        --width=576 \
        --height=320 \
        --learning_rate=1e-5 \
        --per_gpu_batch_size=1 \
        --num_train_epochs=6 \
        --mixed_precision="fp16" \
        --gradient_accumulation_steps=2 \
        --checkpointing_steps=10000 \
        --validation_steps=1000 \
        --gradient_checkpointing \
        --num_validation_images=1 \
        --checkpoints_total_limit=2 \
        --images_bbox=True \
        --filter_num=5

CUDA_VISIBLE_DEVICES=1 accelerate launch train_svd_traj_blender_14.py \
        --pretrained_model_name_or_path="../stable-video-diffusion-img2vid" \
        --output_dir="model_out_objaverse_10k_ft_long" \
        --video_folder="../render_objaverse_10k" \
        --validation_image_folder="../dataset/objaverse_val" \
        --width=576 \
        --height=320 \
        --learning_rate=1e-5 \
        --per_gpu_batch_size=1 \
        --num_train_epochs=6 \
        --mixed_precision="fp16" \
        --gradient_accumulation_steps=2 \
        --checkpointing_steps=10000 \
        --validation_steps=1000 \
        --gradient_checkpointing \
        --num_validation_images=1 \
        --checkpoints_total_limit=2 \
        --controlnet_model_name_or_path=model_out_objaverse_10k_bbox_long/checkpoint-50000 \
        --filter_num=5
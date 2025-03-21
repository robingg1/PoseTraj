# PoseTraj
### [CVPR 2025] PoseTraj: Pose-Aware Trajectory Control in Video Diffusion

## Steps for rendering PoseTraj-10k
### Step 1: Install blender
### Step 2: Download Objaverse dataset.
Following instruction of [Objaverse Codebase](https://github.com/allenai/objaverse-xl)。
### Step 3: Write filelist into rendered_split.txt
### Step 4: Render videos
```
./blender --background --python render_objverse/render_objs_v2.py
```
change objaverse path, filelist path, start idx, end idx in the script.
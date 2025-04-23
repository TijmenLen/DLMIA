import nibabel as nib
import numpy as np
import pyvista as pv
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
import os
import matplotlib.animation as animation
import imageio_ffmpeg
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r"C:\Users\User\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"  # Replace with the actual path

print(imageio_ffmpeg.get_ffmpeg_exe())

def get_voxel_size(img_path):
    nimg = nib.load(img_path)
    voxel_sizes = nimg.header.get_zooms()[:3]  # Get the first three dimensions
    return voxel_sizes

def visualize_4d_image(image_path, output_video_path):
    # Load the 4D image
    img = nib.load(image_path)
    data = img.get_fdata()

    if len(data.shape) != 4:
        raise ValueError("The provided image is not 4D. Please provide a 4D image.")

    # Extract dimensions
    x_dim, y_dim, z_dim, t_dim = data.shape

    voxel_sizes = get_voxel_size(image_path)

    # Create a PyVista plotter with a scaled window size
    plotter = pv.Plotter(off_screen=True)  

    # Create a writer for the video
    writer = imageio.get_writer(output_video_path, fps=2)  # 2 frames per second

    # Define a custom colormap for segmentation masks
    custom_cmap = ListedColormap(["white", "red", "green", "blue"])  # Colors for mask 1, 2, and 3

    # Function to update the plot for each time frame

    def update(frame):
        plotter.clear()
        plotter.add_axes()
        plotter.add_text(f"Time Frame {frame + 1}/{t_dim}", font_size=12, color="black")

        # Extract the 3D slice for the current time frame
        slice_3d = data[:, :, :, frame]

        # Create a PyVista ImageData grid
        grid = pv.ImageData()
        grid.dimensions = slice_3d.shape
        grid.spacing = voxel_sizes  # Adjust spacing if needed
        grid.point_data["values"] = slice_3d.flatten(order="F")

        # Add the grid as a voxel representation
        plotter.add_volume(grid, cmap=custom_cmap, opacity="foreground")  # Removed `norm` and set full opacity

        # Set the camera to zoom into the middle of the image
        center_x, center_y, center_z = (x_dim * voxel_sizes[0] / 2, 
                                        y_dim * voxel_sizes[1] / 2, 
                                        z_dim * voxel_sizes[2] / 2)
        plotter.camera_position = [
            (center_x + 500, center_y + 500, center_z + 500),  # Camera position (oblique angle)
            (center_x, center_y, center_z),                   # Focal point (center of the image)
            (0, 0, 1)                                         # View up direction
        ]

        # Render the frame and save it to the video
        img = plotter.screenshot(return_img=True)
        writer.append_data(img)

    # Loop through each time frame
    for frame in range(t_dim):
        update(frame)

    # Close the writer
    writer.close()
    print(f"Video saved to {output_video_path}")


def visualize_mri_with_segmentation_matplotlib(image_path, segmentation_path, slice_positions):

    # Load the 4D MRI image
    mri_img = nib.load(image_path)
    mri_data = mri_img.get_fdata()

    # Load the 4D segmentation mask
    seg_img = nib.load(segmentation_path)
    seg_data = seg_img.get_fdata()

    if len(mri_data.shape) != 4 or len(seg_data.shape) != 4:
        raise ValueError("Both MRI and segmentation images must be 4D.")

    # Extract the first time frame (time frame 0)
    mri_first_frame = mri_data[:, :, :, 0]
    seg_first_frame = seg_data[:, :, :, 0]

    # Exclude the first mask (assume label 0 is the first mask)
    seg_first_frame[seg_first_frame == 0] = np.nan  # Set label 0 to NaN for transparency

    # Retrieve voxel sizes from the NIfTI header
    voxel_sizes = mri_img.header.get_zooms()[:3]

    # Define slice positions
    x_slice, y_slice, z_slice = slice_positions

    # Create a GridSpec layout
    fig = plt.figure(figsize=(16,6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,1], width_ratios=[1, 1.5])  # Adjust width_ratios

    # Define a custom colormap for segmentation masks
    custom_cmap = ListedColormap(["red", "green", "blue"])  # Colors for mask 1, 2, and 3
    bounds = [0.5, 1.5, 2.5, 3.5]  # Define boundaries for mask labels (1, 2, 3)
    norm = BoundaryNorm(bounds, custom_cmap.N)  # Map labels to colors

    # Axial view (Z-axis slice) spanning the height of both rows
    ax_axial = fig.add_subplot(gs[:, 0])  # Span both rows in the first column
    ax_axial.imshow(mri_first_frame[:, :, z_slice], cmap="gray", aspect=voxel_sizes[0] / voxel_sizes[1])
    ax_axial.imshow(seg_first_frame[:, :, z_slice], cmap=custom_cmap, norm=norm, alpha=0.5, aspect=voxel_sizes[0] / voxel_sizes[1])
    ax_axial.axhline(y=x_slice, color="yellow", linestyle="--", linewidth=1.5)  # Coronal axis
    ax_axial.axvline(x=y_slice, color="yellow", linestyle="--", linewidth=1.5)  # Sagittal axis
    ax_axial.set_title(f"Segmentation - Axial View (Z={z_slice})")
    ax_axial.axis("off")

    # Coronal view (Y-axis slice) in the top-right corner
    ax_coronal = fig.add_subplot(gs[0, 1])
    rotated_coronal_mri = np.rot90(mri_first_frame[:, y_slice, :], k=-1)
    rotated_coronal_seg = np.rot90(seg_first_frame[:, y_slice, :], k=-1)
    ax_coronal.imshow(rotated_coronal_mri, cmap="gray", aspect=voxel_sizes[2] / voxel_sizes[0])
    ax_coronal.imshow(rotated_coronal_seg, cmap=custom_cmap, norm=norm, alpha=0.5, aspect=voxel_sizes[2] / voxel_sizes[0])
    ax_coronal.axhline(y=z_slice-0.5, color="yellow", linestyle="--", linewidth=1.5)  # Axial axis
    ax_coronal.set_title(f"Segmentation - Coronal View (Y={y_slice})")
    ax_coronal.axis("off")

    # Sagittal view (X-axis slice) in the bottom-right corner
    ax_sagittal = fig.add_subplot(gs[1, 1])
    ax_sagittal.imshow(mri_first_frame[x_slice, :, :].T, cmap="gray", aspect=voxel_sizes[2] / voxel_sizes[1])
    ax_sagittal.imshow(seg_first_frame[x_slice, :, :].T, cmap=custom_cmap, norm=norm, alpha=0.5, aspect=voxel_sizes[2] / voxel_sizes[1])
    ax_sagittal.axhline(y=z_slice-0.5, color="yellow", linestyle="--", linewidth=1.5)  # Axial axis
    ax_sagittal.set_title(f"Segmentation - Sagittal View (X={x_slice})")
    ax_sagittal.axis("off")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def visualize_mri_with_segmentation_matplotlib(image_path, segmentation_path, slice_positions, output_video_path):

    # Load the 4D MRI image
    mri_img = nib.load(image_path)
    mri_data = mri_img.get_fdata()

    # Load the 4D segmentation mask
    seg_img = nib.load(segmentation_path)
    seg_data = seg_img.get_fdata()

    if len(mri_data.shape) != 4 or len(seg_data.shape) != 4:
        raise ValueError("Both MRI and segmentation images must be 4D.")

    # Retrieve voxel sizes from the NIfTI header
    voxel_sizes = mri_img.header.get_zooms()[:3]

    # Define slice positions
    x_slice, y_slice, z_slice = slice_positions

    # Create a figure for the animation
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.5])

    # Define a custom colormap for segmentation masks
    custom_cmap = ListedColormap(["red", "green", "blue"])  # Colors for mask 1, 2, and 3
    bounds = [0.5, 1.5, 2.5, 3.5]  # Define boundaries for mask labels (1, 2, 3)
    norm = BoundaryNorm(bounds, custom_cmap.N)  # Map labels to colors

    # Axial view (Z-axis slice) spanning the height of both rows
    ax_axial = fig.add_subplot(gs[:, 0])  # Span both rows in the first column
    ax_coronal = fig.add_subplot(gs[0, 1])  # Coronal view
    ax_sagittal = fig.add_subplot(gs[1, 1])  # Sagittal view

    def update(frame):
        # Clear axes
        ax_axial.clear()
        ax_coronal.clear()
        ax_sagittal.clear()

        # Extract the current time frame
        mri_frame = mri_data[:, :, :, frame]
        seg_frame = seg_data[:, :, :, frame]

        # Exclude the first mask (assume label 0 is the first mask)
        seg_frame[seg_frame == 0] = np.nan  # Set label 0 to NaN for transparency

        # Axial view
        ax_axial.imshow(mri_frame[:, :, z_slice], cmap="gray", aspect=voxel_sizes[0] / voxel_sizes[1])
        ax_axial.imshow(seg_frame[:, :, z_slice], cmap=custom_cmap, norm=norm, alpha=0.5, aspect=voxel_sizes[0] / voxel_sizes[1])
        ax_axial.axhline(y=x_slice, color="yellow", linestyle="--", linewidth=1.5)  # Coronal axis
        ax_axial.axvline(x=y_slice, color="yellow", linestyle="--", linewidth=1.5)  # Sagittal axis
        ax_axial.set_title(f"Axial View (Z={z_slice}, Frame={frame + 1})")
        ax_axial.axis("off")

        # Coronal view
        rotated_coronal_mri = np.rot90(mri_frame[:, y_slice, :], k=-1)
        rotated_coronal_seg = np.rot90(seg_frame[:, y_slice, :], k=-1)
        ax_coronal.imshow(rotated_coronal_mri, cmap="gray", aspect=voxel_sizes[2] / voxel_sizes[0])
        ax_coronal.imshow(rotated_coronal_seg, cmap=custom_cmap, norm=norm, alpha=0.5, aspect=voxel_sizes[2] / voxel_sizes[0])
        ax_coronal.axhline(y=z_slice - 0.5, color="yellow", linestyle="--", linewidth=1.5)  # Axial axis
        ax_coronal.set_title(f"Coronal View (Y={y_slice}, Frame={frame + 1})")
        ax_coronal.axis("off")

        # Sagittal view
        ax_sagittal.imshow(mri_frame[x_slice, :, :].T, cmap="gray", aspect=voxel_sizes[2] / voxel_sizes[1])
        ax_sagittal.imshow(seg_frame[x_slice, :, :].T, cmap=custom_cmap, norm=norm, alpha=0.5, aspect=voxel_sizes[2] / voxel_sizes[1])
        ax_sagittal.axhline(y=z_slice - 0.5, color="yellow", linestyle="--", linewidth=1.5)  # Axial axis
        ax_sagittal.set_title(f"Sagittal View (X={x_slice}, Frame={frame + 1})")
        ax_sagittal.axis("off")

    # Create an animation
    ani = animation.FuncAnimation(fig, update, frames=mri_data.shape[3], interval=500)

    # Use FFMpegWriter explicitly
    ffmpeg_writer = FFMpegWriter(fps=2, metadata=dict(artist="Matplotlib"), bitrate=1800)

    # Save the animation as a video
    ani.save(output_video_path, writer=ffmpeg_writer)
    print(f"Matplotlib video saved to {output_video_path}")

if __name__ == "__main__":
    # Define the input and output folders
    image_folder = r"4D_image"
    segmentation_folder = r"4D_segmentation"
    video_save_folder = r"4D_video"

    # Ensure the output folder exists
    os.makedirs(video_save_folder, exist_ok=True)

    # Loop through all files in the image folder
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(".nii.gz"):  # Process only NIfTI files
            # Construct the full paths for the image and segmentation
            image_path = os.path.join(image_folder, image_filename)
            segmentation_filename = f"segmentation_{image_filename}"  # Assuming segmentation files follow this naming convention
            segmentation_path = os.path.join(segmentation_folder, segmentation_filename)

            # Check if the corresponding segmentation file exists
            if not os.path.exists(segmentation_path):
                print(f"Segmentation file not found for {image_filename}. Skipping...")
                continue

            # Dynamically create the video paths
            base_name = os.path.splitext(image_filename)[0]  # Remove the extension
            video_name = f"video_{base_name}.mp4"
            video_path = os.path.join(video_save_folder, video_name)
            matplotlib_video_name = f"video_matplotlib_{base_name}.mp4"
            matplotlib_video_path = os.path.join(video_save_folder, matplotlib_video_name)

            # Load the MRI image to get its dimensions
            mri_img = nib.load(image_path)
            mri_data = mri_img.get_fdata()
            x_dim, y_dim, z_dim, _ = mri_data.shape

            # Define slice positions as half of the image dimensions
            slice_positions = (x_dim // 2, y_dim // 2, z_dim // 2)

            # Call the functions to create the videos
            print(f"Processing {image_filename}...")
            visualize_4d_image(segmentation_path, video_path)
            visualize_mri_with_segmentation_matplotlib(image_path, segmentation_path, slice_positions, matplotlib_video_path)
            print(f"Finished processing {image_filename}.")
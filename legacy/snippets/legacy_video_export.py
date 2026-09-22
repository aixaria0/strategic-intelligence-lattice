# HISTORICAL FRAGMENT: original export produced frames from dummy, not measured, surfaces.
# Plotly + kaleido + imageio/ffmpeg were proposed optional dependencies.
# Frame rendering inside the simulation loop increases latency and memory use.
import os
import imageio
import plotly.graph_objects as go

frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

def export_frame(fig: go.Figure, index: int) -> str:
    path = os.path.join(frames_dir, f"frame_{index:03d}.png")
    fig.write_image(path, scale=2)
    return path

def export_video(frame_paths, video_path="entropy_basin_tournament.mp4", fps=2):
    with imageio.get_writer(video_path, fps=fps) as writer:
        for path in frame_paths:
            writer.append_data(imageio.imread(path))
    return video_path

# Retained as an optional visualization concept, not an entropy or research result.

from moviepy.editor import VideoFileClip
from moviepy.video.fx import all as vfx
def speedup_and_resize(input_path, output_path, speed_factor=6.0, scale_factor=2.0):
    # 加载视频
    clip = VideoFileClip(input_path)

    # 加速（时间缩短）
    sped_up = clip.fx(vfx.speedx, factor=speed_factor)

    # 分辨率缩放
    resized = sped_up.resize(scale_factor)

    # 输出新视频
    resized.write_videofile(output_path, codec='libx264', audio_codec='aac')

# 示例调用
if __name__ == "__main__":
    speedup_and_resize(
        input_path="/home/ubuntu/Pictures/Picturesfeicuwian_long_video1.mp4",
        output_path="/home/ubuntu/Pictures/feicuwian_long_video1.mp4",
        speed_factor=1.0,
        scale_factor=2.0
    )

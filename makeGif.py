import glob
from PIL import Image


def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.JPG")]
    frame_one = frames[0]
    frame_one.save("results_animation.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


if __name__ == "__main__":
    path=r"E:\KEYPOINT_DETECTOR\Keypoint_detec_results_groundTruth_predictions_validation"
    make_gif(path)
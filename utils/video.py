"""
Utility functions for videos, requires opencv
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import glob
import multiprocessing
import os
import subprocess
import sys
from tqdm import tqdm


def extract_frames(video_path, frames_dir=None, overwrite=False, start=-1, end=-1, every=1, seconds=False):
    """
    Extract frames from a video using OpenCVs VideoCapture

    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :param seconds: is the start and finish in frames or seconds
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    if frames_dir is not None:
        frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV
    if seconds:
        start = int(start*capture.get(cv2.CAP_PROP_FPS))
        end = int(end*capture.get(cv2.CAP_PROP_FPS))

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    os.makedirs(frames_dir, exist_ok=True)
    out_frames = list()
    while frame < end:  # lets loop through the frames until the end

        ret, image = capture.read()  # read an image from the capture

        if while_safety > 10:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if ret == 0 or image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            if frames_dir is not None:
                # save in start of chunk subdirectory in video name subdirectory
                save_path = os.path.join(frames_dir, "{:010d}.jpg".format(frame))
                if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                    cv2.imwrite(save_path, image)  # save the extracted image
                    saved_count += 1  # increment our counter by one
            else:
                out_frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture
    if frames_dir is not None:
        return saved_count  # and return the count of the images we saved
    else:
        return out_frames


def video_to_frames(video_path, frames_dir, stats_dir=None, overwrite=False, every=1, chunk_size=1000):
    """
    Extracts the frames from a video using multiprocessing

    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param stats_dir: directory to store video stats .txt files
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    if not os.path.exists(video_path):
        print("Video doesn't exist: %s" % video_path)
        return None

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # get its frame width
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # get its frame height
    fps = int(capture.get(cv2.CAP_PROP_FPS))  # get its frame height
    capture.release()  # release the capture straight away

    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV installation.")
        return None  # return None

    frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)  # make sure last chunk has correct end frame

    frames_dir = os.path.join(frames_dir, video_filename)

    for frame_chunk in frame_chunks:
        # make directory to save frames, its a sub dir in the frames_dir with the video name
        # also since file systems hate lots of files in one directory, lets put separate chunks in separate directories
        frames_dir = os.path.join(frames_dir, "{:010d}".format(frame_chunk[0]))

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [executor.submit(extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)

    if stats_dir is not None:  # if we specify a stats directory
        os.makedirs(stats_dir, exist_ok=True)  # make the directory if it doesn't already exist
        with open(os.path.join(stats_dir, video_filename + '.txt'), 'w') as f:  # make a file
            f.write('{},{},{},{},{}'.format(video_filename, width, height, total, fps))  # write out the video stats

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames


def frames_to_video(frames_dir, video_path, fps=30):
    """
    Generates a .mp4 video from a directory of frames

    :param frames_dir: the directory containing the frames, note that this and any subdirs be looked through recursively
    :param video_path: path to save the video
    :param fps: the frames per second to make the output video
    :return: the output video path, or None if error
    """

    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible
    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible

    # add the .mp4 extension if it isn't already there
    if video_path[-4:] != ".mp4":
        video_path += ".mp4"

    # get the frame file paths
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        files = glob.glob(frames_dir + "/**/*" + ext, recursive=True)
        if len(files) > 0:
            break

    # couldn't find any images
    if not len(files) > 0:
        print("Couldn't find any files in {}".format(frames_dir))
        return None

    # get first file to check frame size
    image = cv2.imread(files[0])
    height, width, _ = image.shape  # need to get the shape of the frames

    # sort the files alphabetically assuming this will do them in the correct order
    files.sort()

    # create the videowriter - will create an .mp4
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

    # load and write the frames to the video
    for filename in tqdm(files, desc="Generating Video {}".format(video_path)):
        image = cv2.imread(filename)  # load the frame
        video.write(image)  # write the frame to the video

    video.release()  # release the video

    return video_path


def download_youtube(save_dir, yt_id, v_id=None):
    """
    download a video from youtube
    requires youtube-dl

    :param save_dir: the directory to save the vids
    :param yt_id: the youtube video id
    :param v_id: the video id for saving, if not specified will be the yt_id
    :return: the name.ext as a string or None if download was unsuccessful
    """

    if v_id is None:
        v_id = yt_id

    extensions = [".mp4", ".mkv", ".mp4.webm"]

    # check if it exists first
    for ext in extensions:
        if os.path.exists(os.path.join(save_dir, v_id + ext)):
            return v_id + ext

    os.makedirs(save_dir, exist_ok=True)
    subprocess.run(["youtube-dl -o '" + os.path.join(save_dir, v_id + ".mp4") + "' 'http://youtu.be/" + yt_id + "'"
                    + " --quiet --no-warnings --ignore-errors "], shell=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # check we have downloaded it
    for ext in extensions:
        if os.path.exists(os.path.join(save_dir, v_id + ext)):
            return v_id + ext

    # if didn't work will return None
    return None

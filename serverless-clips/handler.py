import cv2
from easyocr import Reader
import boto3
import requests

import os
import json
from datetime import datetime

from configparser import ConfigParser

config = ConfigParser()
config.read('config.cfg')


"""
This app will load a S3 video link from a surveilance camera and create a clip around a certain time stamp.
To find the right time for the clip the timestamp watermark is read in the image using computer vision text
extraction and then used to find the right frames for clipping.
To avoid having to extract the timestamp in each frame only the first and last frame is used and then the 
frames for video start and end are calculated using the start and end time of the movie and the FPS.
The clip is then created with the given seconds prior and after.
Eg a clip requested at event 22:10:50 with 2 seconds prior and 5 after will create a 7s long clip starting at
22:10:48 and ending at 22:10:57 with a marker in the clip at the event at 22:10:50.
"""


def download_to_local(video_link, LOCAL_TEMP_DIR):
    """
    fetch S3 file to local temp dir
    
    :param video_link: S3 link
    :param LOCAL_TEMP_DIR: local temp dir
   
    """
    print("Downloading video from s3...")

    local_file_name = video_link.split('/')[-1]
    local_file_path = os.path.join(LOCAL_TEMP_DIR, local_file_name)

    r = requests.get(video_link, stream=True)

    # download in chunks
    with open(local_file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    file_size = os.path.getsize(local_file_path)
    if file_size < 1000:
        raise Exception("Video file on S3 cannot be fetched : {}".format(video_link))

    video_name = local_file_name.split(".")[0]
    print("Downloaded S3 video to {}".format(local_file_path))
    return local_file_path, video_name


def get_image_count(path_video):
    """
    To know how where to set the start and end frame the number of frames and fps is used to calcuate the
    right frames.
    :param path_video: movie file
   
    """
    cap = cv2.VideoCapture(path_video)
    # get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return totalFrames, fps


def extract_first_and_last_frame(path_video, LOCAL_TEMP_DIR):
    """
    First and last frame is used to get the time span the video covers to compute the event frame.
    :param path_video: movie file
    :param LOCAL_TEMP_DIR: local temp dir
   
    """
    print("Extracting first and last frame from whole video...")
    cap = cv2.VideoCapture(path_video)

    # get first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # read frame
    has_frames, frame = cap.read()
    # save frame as JPG file
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    frame_save_path_first = os.path.join(LOCAL_TEMP_DIR, "image_first" + ".jpg")
    cv2.imwrite(frame_save_path_first, frame)

    # get last frame
    # get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, totalFrames-1)
    has_frames, frame = cap.read()
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    frame_save_path_last = os.path.join(LOCAL_TEMP_DIR, "image_last" + ".jpg")
    cv2.imwrite(frame_save_path_last, frame)

    cap.release()
    return frame_save_path_first, frame_save_path_last


def extract_specific_frame(path_video, LOCAL_TEMP_DIR, frame_number, from_end=False):
    """
    
    :param path_video: movie file
    :param LOCAL_TEMP_DIR: local temp dir
    :param frame_number: the frame to extract
    :param from_end: if we work with the end of the movie, move backwards if image failed to OCR
   
    """
    print("Extracting first and last frame from whole video...")
    cap = cv2.VideoCapture(path_video)

    # get total number of frames
    if from_end:
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_number = totalFrames - frame_number

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frames, frame = cap.read()
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    if from_end:
        frame_save_path = os.path.join(LOCAL_TEMP_DIR, "image_last" + ".jpg")
    else:
        frame_save_path = os.path.join(LOCAL_TEMP_DIR, "image_first" + ".jpg")
    cv2.imwrite(frame_save_path, frame)

    cap.release()
    return frame_save_path


def check_if_date_or_time(text):
    """
    Make sure the right timestamp was extracted by OpenCV. If the data can't be parsed into time format,
    take the next frame and try again, as the timestamp might be covered in image noise.
    :param text: text from OCR to check
   
    """
    # remove whitespace
    text = text.replace(" ", "")
    text = text.replace("_", "-")
    print(text)
    try:
        date = datetime.strptime(text, '%Y-%m-%d')
        return 'date', date
    except Exception:
        pass
    try:
        time = datetime.strptime(text, "%H:%M:%S").time()
        return 'time', time
    except Exception:
        pass
    return 'None', None


def extract_time(frame_save_path, LOCAL_TEMP_DIR):
    """
    Use Open CV OCR to extract text from the image
    :param frame_save_path: local saved frame to OCR from
    :param LOCAL_TEMP_DIR: local temp dir
   
    """
    print("Extracting times stamps from video...")
    # load the input image from disk
    image = cv2.imread(frame_save_path)
    # OCR the input image using EasyOCR
    print("[INFO] OCR'ing input image...")
    reader = Reader(['en'], gpu=-1)
    results = reader.readtext(image)
    date_for_save = "nd"
    time_for_save = "nt"
    date_read = None
    time_read = None
    # loop over the results
    for (bbox, text, prob) in results:
        # display the OCR'd text and associated probability
        print("[INFO] {:.4f}: {}".format(prob, text))
        # unpack the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        # check if date or time
        type, time_date = check_if_date_or_time(text)
        if type == 'date':
            date_read = time_date
            date_for_save = time_date.strftime('%Y-%m-%d')
            print('Date Read')
        if type == 'time':
            time_for_save = text.replace(":", "_")
            time_read = time_date
            print('Time Read')

        # with the OCR'd text itself
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(image, text, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # show the output image
    frame_save_path = os.path.join(LOCAL_TEMP_DIR, "image" + "_" + date_for_save + "_" + time_for_save + ".jpg")
    cv2.imwrite(frame_save_path, image)

    # for testing show image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    return date_read, time_read


def get_frames_to_clip(first_date_stamp, last_date_stamp, event_date_stamp, seconds_prior, seconds_after, time_offset_forward, time_offset_backward, fps):
    """
    Based on the stand and end time of the whole movie, calculate the clip start, end and event frame
    :param first_date_stamp: start of movie
    :param last_date_stamp: end of movie
    :param event_date_stamp: time of event
    :param seconds_prior: how many seconds to include before event
    :param seconds_after: how many seconds to include after event
    :param time_offset_forward: if failed to extract time at start of movie and offset is used, consider here
    :param time_offset_backward: if failed to extract time at end of movie and offset is used, consider here
    :param fps: fps
   
    """
    print("Caculating frames to clip..")
    if event_date_stamp < first_date_stamp or event_date_stamp > last_date_stamp:
        raise Exception("Requested time stamp {} not in video (start: {} - end: {}".format(event_date_stamp, first_date_stamp, last_date_stamp))

    # get frame of event
    # get time where event happened in relation to start of video, if the timestamp was not at start due to noise, correct by adding 1s
    if time_offset_forward:
        event_seconds = (event_date_stamp - first_date_stamp).seconds + 1
    else:
        event_seconds = (event_date_stamp - first_date_stamp).seconds
    # convert to frame number
    event_frame = int(event_seconds * fps)

    # get time where event happened minus prior time requested in relation to start of video
    requested_clip_start_seconds = event_seconds - seconds_prior
    requested_clip_start_frame = int(requested_clip_start_seconds * fps)

    if requested_clip_start_frame < 0:
        requested_clip_start_frame = 0
        print('Not enough time for {}s prior event at {}s, beginning clip at start of video'.format(seconds_prior, event_seconds))

    # get time where event happened plus after time requested in relation to start of video
    if time_offset_backward:
        requested_clip_end_seconds = event_seconds + seconds_after - 1
    else:
        requested_clip_end_seconds = event_seconds + seconds_after
    requested_clip_end_frame = int(requested_clip_end_seconds * fps)

    frames_in_video = int((last_date_stamp - first_date_stamp).seconds * fps)
    if requested_clip_end_frame > frames_in_video:
        requested_clip_end_frame = frames_in_video
        print('Not enough time for {}s after event at {}s, ending clip at end of video'.format(seconds_after, event_seconds))

    return event_frame, requested_clip_start_frame, requested_clip_end_frame


def cutout_clip(path_video, event_frame, start_frame, end_frame, fps, video_name, LOCAL_TEMP_DIR):
    """
    Create the clip based on the frames identified
    
    :param path_video: movie file
    :param event_frame: event frame
    :param start_frame: start frame
    :param end_frame: end frame
    :param fps: fps
    :param video_name: save name of clip
    :param LOCAL_TEMP_DIR: local dir
   
    """
    print("Clipping video between frames {} and {}".format(start_frame, end_frame))
    # open whole video as source
    cap = cv2.VideoCapture(path_video)
    has_frames, frame = cap.read()
    height, width, _ = frame.shape

    # create writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    file_name = "{}_clipped_{}.avi".format(video_name, event_frame)
    local_save_path = os.path.join(LOCAL_TEMP_DIR, file_name)
    clip_writer = cv2.VideoWriter(local_save_path, fourcc, fps, (width, height))

    # as long as the whole video has frames
    frame_index = 0
    while has_frames:
        frame_index += 1
        # if frame index is between start and end, write to clip writer
        if frame_index >= start_frame and frame_index <= end_frame:
            # if we are at the event frame and the 20 frames after, draw rectangle in top right corner to note event
            if frame_index >= event_frame and frame_index <= event_frame + 20:
                cv2.rectangle(frame, (0, 0), (50, 50), color=(0, 255, 0), thickness=3)
            clip_writer.write(frame)
        has_frames, frame = cap.read()
    cap.release()
    clip_writer.release()

    print("Clipping done, saved locally under: {}".format(local_save_path))
    return local_save_path, file_name


def upload_clip_to_s3(save_path, file_name):
    """
    Upload ready clip to S3
    :param save_path: local clip path
    :param file_name: name to use on S3
   
    """
    # save to S3
    print("Uploading to S3: {}".format(save_path))

    session = boto3.Session(
        aws_access_key_id=config['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=config['AWS_SECRET_ACCESS_KEY'],
    )

    bucket = "clips"
    s3_key = "temp_clips/" + file_name

    s3 = session.resource('s3')

    s3.meta.client.upload_file(Filename=save_path, Bucket=bucket, Key=s3_key, ExtraArgs={'ACL': 'public-read'})
    download_link = "https://" + bucket + ".s3.amazonaws.com/" + s3_key

    return download_link


def create_event_clip(event, context):
    """
    Main serverless function to handle the clip creation.
    :param event: what is passed from API as event needed
    :param context: serverless context (not used)
   
    """
    # parse API parameters
    parameters = event["queryStringParameters"]
    video_link = parameters['video_link']
    event_date = parameters['event_date']
    event_time = parameters['event_time']
    seconds_prior = int(parameters['seconds_prior'])
    seconds_after = int(parameters['seconds_after'])

    LOCAL_TEMP_DIR = config['local_temp_dir'] 
    
    try:
        # check event args are good
        try:
            event_date_stamp = datetime.combine(datetime.strptime(event_date, '%Y-%m-%d').date(), datetime.strptime(event_time, "%H:%M:%S").time())
        except ValueError:
            raise Exception("Please supply event date in form YYYY-MM-DD and event time in HH:MM:SS")

        # fetch to local
        path_video, video_name = download_to_local(video_link, LOCAL_TEMP_DIR)

        # first get image count so we can check time of first and last image
        image_count, fps = get_image_count(path_video)

        # get first and last frames from video
        frame_save_path_first, frame_save_path_last = extract_first_and_last_frame(path_video, LOCAL_TEMP_DIR)

        # get first and last time stamp from video
        time_offset_backward = False
        time_offset_forward = False
        date_read_first, time_read_first = extract_time(frame_save_path_first, LOCAL_TEMP_DIR)

        # if failed to read, go one second frames forward and try again, as might be obstructed in image
        if date_read_first == None or time_read_first == None:
            frame_save_path_first = extract_specific_frame(path_video, LOCAL_TEMP_DIR, int(fps), from_end=False)
            date_read_first, time_read_first = extract_time(frame_save_path_first, LOCAL_TEMP_DIR)
            # save that the start stamp is one second later than end of frame
            time_offset_forward = True
        #if still not clear date, fail
        if date_read_first == None or time_read_first == None:
            raise Exception('Start Datestamp could not be read from file: {}'.format(frame_save_path_first))
        first_date_stamp = datetime.combine(date_read_first, time_read_first)

        date_read_last, time_read_last = extract_time(frame_save_path_last, LOCAL_TEMP_DIR)
        # if failed to read, go one second frames backward and try again, as might be obstructed in image
        if date_read_last == None or time_read_last == None:
            frame_save_path_last = extract_specific_frame(path_video, LOCAL_TEMP_DIR, int(fps), from_end=True)
            date_read_last, time_read_last = extract_time(frame_save_path_last, LOCAL_TEMP_DIR)
            # save that the end stamp is one second earlier than end of frame
            time_offset_backward = True
        if date_read_last == None or time_read_last == None:
            raise Exception('End Datestamp could not be read from file: {}'.format(frame_save_path_first))
        last_date_stamp = datetime.combine(date_read_last, time_read_last)

        # get the frame of event, start_frame and end_frame requested
        event_frame, start_frame, end_frame = get_frames_to_clip(first_date_stamp, last_date_stamp, event_date_stamp, seconds_prior, seconds_after, time_offset_forward, time_offset_backward, fps)

        # clip the main video around event
        local_save_path, clip_name = cutout_clip(path_video, event_frame, start_frame, end_frame, fps, video_name, LOCAL_TEMP_DIR)

        s3_link = upload_clip_to_s3(local_save_path, clip_name)

        response = {
            'statusCode': 200,
            'body': json.dumps({'clip_link': s3_link})
        }
        return response

    except Exception as e:
        status_code = 400
        return {
            'statusCode': status_code,
            'body': json.dumps(str(e))
        }

# for local testing
if __name__ == "__main__":
    video_link = "S3bucket/file.mp4"
    seconds_prior = 1
    seconds_after = 3
    event_date = '2022-08-11'
    event_time = '00:06:37'

    print(create_event_clip(video_link, event_date, event_time, seconds_prior, seconds_after))



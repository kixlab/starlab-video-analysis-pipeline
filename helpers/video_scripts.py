import os

import json

import cv2

import re

from yt_dlp import YoutubeDL

from helpers import transcribe_audio, segment_into_sentences

DATABASE = "static/database"

def download_video(video_link):
    # Download video 480p or, if short, whatever is available
    options = {
        'format': 'bv[height<=?480][ext=mp4]+ba[ext=mp3]/best',
		#'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=mp3]/best',
        'outtmpl': os.path.join(DATABASE, '%(id)s.%(ext)s'),
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': {'en'},  # Download English subtitles
        'subtitlesformat': '/vtt/g',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'keepvideo': True,
        'skip_download': False,
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(video_link, download=False)
        metadata = ydl.sanitize_info(info)
        video_title = metadata.get('id')
        video_path = os.path.join(DATABASE, f'{video_title}.mp4')
        if not os.path.exists(video_path):
            ydl.download([video_link])
            print(f"Video '{video_title}' downloaded successfully.")
        else:
            print(f"Video '{video_title}' already exists in the directory.")
        return metadata

def extract_frames(video_path):
    frame_paths = []
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' does not exist.")
        return frame_paths

    if not os.path.exists(f"{video_path}_frames"):
        os.makedirs(f"{video_path}_frames")
    else:
        print(f"Frames for video '{video_path}' already exist.")
        for file in os.listdir(f"{video_path}_frames"):
            frame_paths.append(f"{video_path}_frames/{file}")
        frame_paths = sorted(frame_paths, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        return frame_paths
    video_cap = cv2.VideoCapture(video_path)

    ### save frame at each second
    seconds = 0
    while True:
        video_cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000 + 500)
        res, frame = video_cap.read()
        if (res == False):
            break
        frame_path = f"{video_path}_frames/{seconds}.jpg"
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        seconds += 1

    video_cap.release()
    return frame_paths

def extract_transcript_from_audio_openai(audio_path):
    granularity = ["segment"]
    output_path = audio_path.replace(".mp3", f".{'_'.join(granularity)}.json")
    response = None
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            response = json.load(f)
    else:
        response = transcribe_audio(audio_path, granularity)
        with open(output_path, 'w') as f:
            json.dump(response, f, indent=2)
    
    if response is None:
        return []
    
    for segment in response["segments"]:
        segment["text"] = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\']", "", segment["text"])
    
    full_transcript = "".join([segment["text"] for segment in response["segments"]])
    sentences = segment_into_sentences(full_transcript)

    transcript = []
    segment_idx = 0
    line = ""
    start = response["segments"][0]["start"]
    for sentence in sentences:
        sentence = sentence.strip()
        while sentence not in line and segment_idx < len(response["segments"]):
            line += response["segments"][segment_idx]["text"]
            segment_idx += 1
        if sentence not in line:
            print(f"Sentence '{sentence}' not found in the transcript.")
            print(line)
            continue
        finish = response["segments"][segment_idx - 1]["end"]
        start_index = line.index(sentence)
        finish_index = start_index + len(sentence)
        ### do linear interpolation to find the start and finish times for the sentence
        transcript.append({
            "start": start + (finish - start) * start_index / len(line),
            "finish": start + (finish - start) * finish_index / len(line),
            "text": sentence,
        })
        if finish_index < len(line):
            line = line[finish_index:]
            start = transcript[-1]["finish"]
        else:
            line = ""
            if segment_idx < len(response["segments"]):
                start = response["segments"][segment_idx]["start"]
            else:
                start = finish
        
    return transcript

def process_video(video_link):
    video_title = re.split(r"[/=]", video_link)[-1]
    video_path = os.path.join(DATABASE, f'{video_title}.mp4')
    audio_path = os.path.join(DATABASE, f'{video_title}.mp3')

    metadata = download_video(video_link)

    video_frame_paths = extract_frames(video_path)

    subtitles_openai = extract_transcript_from_audio_openai(audio_path)

    return video_title, video_frame_paths, subtitles_openai, metadata

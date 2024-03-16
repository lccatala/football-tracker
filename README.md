# Football player detector
Software capable of analyzing video files of football matches. The end goal is to detect players (distinguishing teams) and referees, and locate the football itself.

The development process and key decisions taken during this development are available in the file `Driblab report Luis Catala.pdf`, along with instructions on how to run the code. For the short version:
1. Create and source a python environment:
`python3 -m venv .venv`
`source .venv/bin/activate`
2. Run the process_video.py script, optionally changing the video and json's output paths:
`python3 process_video.py <input video path> --video_output_path <new video output path> --json_output_path <new json output path>`

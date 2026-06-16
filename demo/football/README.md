# Football demo

Real-time football broadcast overlay: **detection → tracking → overlay**
(`pyml_yolo`/`pyml_objectdetector` -> `pyml_tracker` -> `pyml_football_overlay`).

The overlay draws a foot ellipse per subject (players one colour, referee gold),
a green triangle on the ball, motion trails, and a focal-player HUD + ball contacts + distance.

## Run

```bash
# file -> annotated MP4
demo/football/run.sh
demo/football/run.sh 08fd33_4.mp4 demo/football/out.mp4 1280x720

# live camera -> on-screen
demo/football/run.sh camera /dev/video0
```

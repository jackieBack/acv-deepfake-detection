# Applied Computer Vision - Deep Fake Detection Project

Authors: bb2763@, loo2104@

Course: Applied Computer Vision (COMS 4995)

Semester: Spring 2024

# Step 0. Setting up Cloud Dev + Version Control Environment

Link Github repo with Google Cloud Build, so that we can leverage our GCP instance while doing version control on Github.

# Step 1. Data Collection

Perform data collection of both real and deep fake videos of Joe Biden and Donald Trump. For deep fake videos, utilize Presidential Deep Fake Dataset (https://www.media.mit.edu/publications/presidential-deepfakes-dataset/)

We have organized our dataset as follows:

```
f[real / fake]

--> f[video-clips]: {video-id}.mov / mp3

--> f[text-files]: {video-id}.txt

--> f[screenshots]

  --> f[{video-id}]: {video-id}-{***}.jpg

--> f[audio-files]: {video-id}.wav
```

# Step 2. Fusion Model with Screenshots + Text

Follow the model architecture as laid out in the Midterm report.

It is a early fusion model that feeds in screenshots and transcripted text.

![Model Architecture 1](https://github.com/jackieBack/acv-deepfake-detection/blob/main/assets/model_architecture_1.png)

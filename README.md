# Post-Editing-Interface
With the rapid advancement of artificial intelligence, the demand for intuitive, user-friendly interfaces has increased significantly, particularly among users without programming expertise. At the same time, post-editing has become a key part of sign language technology, allowing users to refine outputs from automatic sign language translation and production systems. Moreover, post-editing is an essential tool for modifying real, human-created sign language data, such as recorded videos or captured poses, to meet specific needs. These considerations inspired the development of a human-centered post-editing interface designed specifically for sign language.

Among the various ways to represent sign language, two significant formats are skeleton poses and videos. Skeleton pose data are typically generated by pose estimation models, which can encounter challenges in certain scenarios. Videos, on the other hand, are either recorded by signers or generated by sign language production systems and often require further adjustments. To address these limitations, this study introduces an interface with two key components: pose editing and video editing.

### Skeleton Pose Editing:
Skeleton pose is a fundamental form of sign language, widely used in sign language recognition, translation, and production. Accurate skeleton pose data enhance the quality of sign language production's output and improve the performance of sign language recognition and translation systems. However, existing pose estimators such as DWPose, OpenPose, AlphaPose, and MediaPipe have notable limitations. For instance, they often fail to detect or accurately locate hand keypoints during rapid movements.

Our system allows users to upload videos or pose data. For videos, users can choose a pose estimator (DWPose, OpenPose, or MediaPipe). Video frames are displayed for selection, and the corresponding pose is shown for manual editing to correct inaccuracies. Updated pose data can then be saved.

[Frame level post editing](https://github.com/user-attachments/assets/f6f4aae4-8883-455b-8bb1-81f670b50dce)


### Video Editing:

This section allows users to modify the original file—whether it’s a pose or video—at the segment
level. If a segment requires adjustment, users can either insert a new video or record one themselves.
The model then synthesizes a new video based on the pose sequence of the inserted or
recorded video and the signer appearance of the original signer.

[Watch the video here](https://github.com/user-attachments/assets/3b32a3e1-3b85-46f9-9380-fe3747106165)

A synthetic video sample:

[Watch the video here](https://github.com/user-attachments/assets/e0cd5997-4a9c-4701-9e6e-9915516a3b74)

Skeleton data:

[Watch the video here](https://github.com/user-attachments/assets/05ce9ef6-486a-4b2f-b929-ed8794b9d2e1)

## Running the Interface

**Installing Packages and Libraries**

Use the following command to install all the necessary packages and libraries:

```bash
pip install -r requirements.txt
```

## Cloning the MusePose Repository

Clone the MusePose repository to your S3IT account using the following command. Since inference with the MusePose model requires a high-end GPU, we recommend running video synthesis on S3IT for optimal performance:

```bash
git clone https://github.com/TMElyralab/MusePose.git
```

Before running the interface, make sure to update the `config.yaml` file by changing the path to your desired directory.

To run the interface, use the following command:

```bash
python main.py
```


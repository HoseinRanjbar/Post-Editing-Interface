# Post-Editing-Interface
Post-editing, a key practice in the translation industry, enables linguists or editors to refine machinetranslated
text, enhancing accuracy and efficiency compared to translating from scratch. This
approach is also valuable in sign language technology, where it can be applied to oversee and improve
the outputs of automatic sign language translation and production systems. 

With the rapid growth of AI, there is an increasing need for more accessible and user-friendly interfaces, especially for users without programming skills. This was another key motivation for us to undertake this project.

The interface is divided into two main components: **Frame-Level Editing** and **Segment-Level Editing**.

### Frame-Level Editing:
In this section, users can manually edit pose data to correct any inaccuracies
from the pose estimator. This functionality is particularly useful for gathering datasets, as it eliminates the need to start from scratch; users can first employ a pose estimator model and then make adjustments. It also aids in evaluating sign pose generation models.
Future updates will add features that enable users to adjust the position,
orientation, and posture of the signer’s body and hands within the frame.

In the video below, you can see some features of the **Frame-Level Editing** component of the interface. The user corrects keypoints that were mistakenly placed by the automatic pose estimator.

[Frame level post editing](https://github.com/user-attachments/assets/f6f4aae4-8883-455b-8bb1-81f670b50dce)


### Segment-Level Editing:

This section allows users to modify the original file—whether it’s a pose or video—at the segment
level. If a segment requires adjustment, users can either insert a new video or record one themselves.
The model then synthesizes a new video based on the pose sequence of the inserted or
recorded video and the signer appearance of the original signer.

[Watch the video here](https://github.com/user-attachments/assets/c80117ad-8504-4d89-99f8-92f3f3b9adeb)

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


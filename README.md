# Post-Editing-Interface
Post-editing, a key practice in the translation industry, enables linguists or editors to refine machinetranslated
text, enhancing accuracy and efficiency compared to translating from scratch. This
approach is also valuable in sign language technology, where it can be applied to oversee and improve
the outputs of automatic sign language translation and production systems. 

With the rapid growth of AI, there is an increasing need for more accessible and user-friendly interfaces, especially for users without programming skills. This was another key motivation for us to undertake this project.

The interface is divided into two main components: **Frame-Level Editing** and **Segment-Level Editing**.

### Frame-Level Editing:
In this section, users can manually edit pose data to correct any inaccuracies
from the pose estimator. Future updates will add features that enable users to adjust the position,
orientation, and posture of the signer’s body and hands within the frame.

In the video below, you can see some features of the **Frame-Level Editing** component of the interface. The user corrects keypoints that were mistakenly placed by the automatic pose estimator.

[Frame level post editing](https://github.com/user-attachments/assets/f6f4aae4-8883-455b-8bb1-81f670b50dce)


### Segment-Level Editing:

This section allows users to modify the original file—whether it’s a pose or video—at the segment
level. If a segment requires adjustment, users can either insert a new video or record one themselves.
The model then synthesizes a new video based on the pose sequence of the inserted or
recorded video and the signer appearance of the original signer.

[Watch the video here](https://github.com/user-attachments/assets/22e214ba-67f0-497c-bed8-0ef9643f8179)


![museppose_gif](https://github.com/user-attachments/assets/7e4f9b37-6c3a-4298-a0d5-ac20356c3262)


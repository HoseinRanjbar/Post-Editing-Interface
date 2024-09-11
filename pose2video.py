import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
background_color = '#464a4d'


def _normalized_to_pixel_coordinates(
    normalized_x, normalized_y, image_width,
    image_height):
  
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def hex_to_bgr(hex_color):
    """Convert a hex color string to a BGR tuple."""
    hex_color = hex_color.lstrip('#')
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    return bgr

def pose2video(pose, face_pose, saving_path, width, height, fps, connections, face_connections, lines_color):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video_writer = cv2.VideoWriter(saving_path, fourcc, fps, (width, height))
    for frame_idx in range(len(pose)):
        # Create a gray frame
        frame_pose = pose[frame_idx]
        face = face_pose[frame_idx]
        #print('frame_pose:{}'.format(frame_pose))
        frame = np.full((height, width, 3), hex_to_bgr(background_color), dtype=np.uint8)    
        # Write the frame to the video file
        frame = draw_lines(frame, frame_pose, face, width, height, connections, face_connections, lines_color)
        for i in range(48):

            color = (245, 47, 86)
            color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            x = frame_pose[0,i]
            y = frame_pose[1,i]

            if x and y <= 1:
                x_px, y_px = _normalized_to_pixel_coordinates(x, y, width, height)
                if x and y != 0:
                    # Draw the circle on the frame
                    cv2.circle(frame, (x_px,y_px), 2, hex_to_bgr(color), 1)

        for j in range(11):
                x = face[0,j]
                y = face[1,j]

                if x and y <= 1:
                    x_px, y_px = _normalized_to_pixel_coordinates(x, y, width, height)
                    if x and y != 0:
                        #self.canvas.create_oval(x_px-4, y_px-4, x_px+4, y_px+4, fill=color, tags="keypoint")
                        cv2.circle(frame, (x_px,y_px), 2, hex_to_bgr(color), 1)

        video_writer.write(frame)

    video_writer.release()

def draw_lines(frame, frame_pose, face, width, height, connections, face_connections, lines_color):
    skeleton = frame_pose
    #print('skeleton{}'.format(skeleton))
    for counter in range(len(connections)):
        
        connection = connections[counter]
        start_idx = connection[0]
        end_idx = connection[1]
        x_start = skeleton[0, start_idx]
        y_start = skeleton[1, start_idx]
        color = lines_color [counter]
        color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        x_end = skeleton[0, end_idx]
        y_end = skeleton[1, end_idx]

        if x_start<1 and x_end<1 and y_end<1 and y_start < 1.0:

            x_start, y_start = _normalized_to_pixel_coordinates(x_start, y_start, width, height)
            x_end, y_end = _normalized_to_pixel_coordinates(x_end, y_end, width, height)
            if (x_end != 0 and y_end != 0) and (x_start != 0 and y_start != 0):
                cv2.line(frame, (x_start,y_start), (x_end, y_end), hex_to_bgr(color), 2)
                
                counter += 1

        for connection in face_connections:

            start_idx = connection[0]
            end_idx = connection[1]

            x_start = face[0, start_idx]
            y_start = face[1, start_idx]
            x_end = face[0, end_idx]
            y_end = face[1, end_idx]

            if x_start<1 and x_end<1 and y_end<1 and y_start < 1.0:
                x_start, y_start = _normalized_to_pixel_coordinates(x_start, y_start, width, height)
                x_end, y_end = _normalized_to_pixel_coordinates(x_end, y_end, width, height)
                cv2.line(frame, (x_start,y_start), (x_end, y_end), hex_to_bgr(color), 2)
            
    return frame


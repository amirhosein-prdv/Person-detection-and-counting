# Person Detection and Counting in Video Footage

This project aims to detect and count the number of persons in a given video footage using computer vision algorithms. The project involves three main steps:

1. Detecting persons in the provided video footage using the RetinaNet algorithm.
2. Implementing a method to accurately count the number of people visible in the entire duration of the video and overlaying this count as a label on the video.
3. Considering a specific zone within the video and developing a process to accurately count the number of people within that zone. The output video should visually represent the specific zone and count persons within that zone.

## Prerequisites
- Python 3.x
- OpenCV (cv2)
- ImageAI library
- Pre-trained YOLOv3 and RetinaNet models

## Installation
1. Clone the repository

2. Install dependencies:

   ```
   pip install opencv-python imageai
   ```

## Usage
1. Ensure you have the video file you want to process.
2. Open `main.ipynb` and run the notebook cells.

## Components
- **Src.py**: This script contains the core functionality for detecting persons in a video and performing the counting tasks.
- **main.ipynb**: This Jupyter notebook serves as the entry point for the project. It defines the YOLOv3 detector for tasks 2 and 3 and the RetinaNet for task 1.
- **models/**: This directory contains pre-trained YOLOv3 and RetinaNet models for person detection.
- **results/**: This directory contains output results.

## Usage Example
```python
from Src import personDetection

# Specify the paths to the input and output video files
video_path_input = "input_video.mp4"
video_path_output = "output_video.mp4"

# Define the polygon zone within the video
polygon = [[250, 80], [380, 80], [380, 160], [250, 160]]

# Perform person detection and counting
personDetection(video_path_input, video_path_output, detector, polygon=polygon, imshow=True)
```

## Approach explanation

1. **Input Preparation**: Prepare the input data, which could be images or video footage containing various objects.

2. **Using Pre-trained Models**: Instead of training a model from scratch, pre-trained models are employed. These models have been previously trained on large datasets, learning to recognize various objects, including persons. By leveraging pre-trained models, we benefit from the knowledge and insights gained during their training, saving time and computational resources. This approach allows us to achieve accurate and reliable object detection results without the need for extensive training on our specific dataset.

3. **Model Inference**:
   - **YOLOv3**: Utilize YOLOv3's single-shot detection approach, which divides the input image into a grid and predicts bounding boxes and class probabilities for each grid cell using a convolutional neural network (CNN).
   - **RetinaNet**: Utilize RetinaNet's feature pyramid network (FPN) to extract multi-scale features and predict objectness scores and class probabilities for anchor boxes at multiple scales and aspect ratios.

4. **Post-processing**: Apply post-processing techniques such as non-maximum suppression (NMS) to filter out redundant and low-confidence detections and retain only the most confident detections.

5. **Output Visualization**: Visualize the detected objects by drawing bounding boxes around them and labeling them with their corresponding class names.

## Explanation of Models

### YOLOv3 (You Only Look Once version 3)

- **Architecture**: YOLOv3 architecture consists of a CNN backbone, typically based on Darknet, followed by detection layers that predict bounding boxes and class probabilities.

![alt text](image.png)

- **Advantages**:
  - Real-time performance suitable for video processing.
  - Simplicity and ease of implementation.
  - Single-pass inference for efficient object detection.
- **Limitations**:
  - May struggle with small objects and heavily occluded objects.
  - Lower precision compared to some two-stage detectors.

### RetinaNet

- **Architecture**: RetinaNet architecture incorporates a feature pyramid network (FPN) to extract multi-scale features and separate classification and regression subnets for predicting objectness scores and bounding box offsets.

![alt text](image-1.png)

- **Advantages**:
  - High accuracy, especially in scenarios with class imbalance.
  - Efficient detection of objects at different scales and aspect ratios.
  - Robustness to small objects and occlusion.
- **Limitations**:
  - Slower inference compared to YOLOv3 due to multi-stage architecture.
  - More complex to implement and train.

## Conclusion

The choice between YOLOv3 and RetinaNet depends on factors such as speed requirements, accuracy demands, and the characteristics of the detection task. YOLOv3 offers real-time performance and simplicity, while RetinaNet provides higher accuracy and robustness to class imbalance and small objects.

### Conclusion

The selection of YOLOv3 and RetinaNet for person detection in the project is based on their well-established performance, real-time capabilities, accuracy, and ease of use. These models offer a good balance between speed and accuracy, making them suitable for processing video footage and accurately detecting persons in various scenarios.



## References
- ImageAI GitHub Repository: [https://github.com/OlafenwaMoses/ImageAI](https://github.com/OlafenwaMoses/ImageAI)
- YOLOv3: An Incremental Improvement: [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)
- Focal Loss for Dense Object Detection: [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)

## License
This project is licensed under the [MIT License](LICENSE).
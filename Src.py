import cv2
import numpy as np

def personDetection(video_path_input, video_path_output, detector, polygon=None, custom=None, imshow=False):
    capture = cv2.VideoCapture(video_path_input)

    if not capture.isOpened():
        print("Error: Couldn't open the video")
        exit()

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec

    output_video = cv2.VideoWriter(video_path_output, fourcc, fps, (width, height))

    while capture.isOpened():
        ret, frame = capture.read()

        if ret:
            
            if polygon is None:
                target_frame = frame
            else:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 255)
                poly_frame = cv2.bitwise_and(frame, frame, mask=mask)
                frame = cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
                target_frame = poly_frame
                

            detections = detector.detectObjectsFromImage(input_image=target_frame,
                                                        minimum_percentage_probability=50,
                                                        custom_objects=custom,)
            
            for person in detections:
                x1, y1, x2, y2 = person['box_points']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            people_count = len(detections)
            cv2.putText(frame, f"People: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            output_video.write(frame)

            if imshow:
                cv2.imshow('Displaying image frames from video file', frame)
        else:
            break

        if cv2.waitKey(25) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
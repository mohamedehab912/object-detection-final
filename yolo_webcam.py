import cvzone
from ultralytics import YOLO
import cv2
import math
detecting ="bottle"
def calculate_horizontal_distance(ref_point, box):
    box_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    distance = ref_point[0] - box_center[0]
    return distance, box_center

def draw_line_to_center(img, ref_point, box_center):
    cv2.line(img, ref_point, box_center, (0, 255, 0), 2)  # Green line
def write_status(img, detecting_status):
    cv2.putText(img, detecting_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


cap = cv2.VideoCapture(0)
model = YOLO("../yolov8_Project/Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]



# Calculate the x-coordinate as the center of the frame
# Calculate the y-coordinate as the top border of the frame (half of the height)
ref_point = (320, 0)  # Assuming a frame size of 640*480

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if classNames[cls] in detecting:
                detecting_status = f'{classNames[cls]} Found!'
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

                # Calculate horizontal distance and get box center
                distance, box_center = calculate_horizontal_distance(ref_point, (x1, y1, x2, y2))
                print(f'Horizontal Distance to {classNames[cls]}: {distance} pixels')

                # Draw line between reference point and box center
                draw_line_to_center(img, ref_point, box_center)
                write_status(img, detecting_status)
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

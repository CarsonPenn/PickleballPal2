from ultralytics import YOLO 


# model = YOLO('models/yolo5-last.pt')
model = YOLO('yolov8x')
# Predict
result = model.predict('input_videos/pball1.mp4',conf=0.2, save=True)
# Track
# result = model.predict('input_videos/pball1.mp4',conf=0.2, save=True)

print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)


# /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 /Users/carsonsmith/Desktop/PickleballPal/yolo_inference.py

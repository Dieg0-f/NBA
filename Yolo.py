from ultralytics import YOLO

model = YOLO('yolov8s')

results = model.predict('video/0.mp4', save=True)

print(results[0])
print("=======================================")
for box in results[0].boxes:
    print(box)
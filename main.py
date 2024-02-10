from ultralytics import YOLO

# Load a model
print("Loading")

model = YOLO("best.pt")

results = model(source=1, show=True, conf=0.4, save=True)

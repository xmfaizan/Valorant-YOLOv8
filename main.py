from ultralytics import YOLO

# Define the task
task = 'detect'

# Initialize the YOLO model with the specified task
model = YOLO("yolov8n.yaml", task=task)

# Training the model
results = model.train(data="data.yaml", epochs=2)

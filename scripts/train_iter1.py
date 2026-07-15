import sys
from pathlib import Path
import shutil

sys.path.insert(0, r"C:\dev\poultry-vision")
from ultralytics import YOLO

data_yaml = r"C:\dev\poultry-vision\dataset\segment\merged_poultry_dataset\data.yaml"
models_dir = Path(r"C:\dev\poultry-vision\models")
models_dir.mkdir(exist_ok=True)

experiments = [
    ("yolov8n", "detect", "yolov8n.pt"),
    ("yolov8s", "detect", "yolov8s.pt"),
    ("yolov12n", "detect", "yolov12n.pt"),
    ("yolov12s", "detect", "yolov12s.pt"),
    ("yolov8n-seg", "segment", "yolov8n-seg.pt"),
    ("yolov8s-seg", "segment", "yolov8s-seg.pt"),
    ("yolov12n-seg", "segment", "yolov12n-seg.pt"),
    ("yolov12s-seg", "segment", "yolov12s-seg.pt"),
]

for name, task, weights in experiments:
    print(f"\n{'='*50}\nStarting Iteration 1 Training: {name} ({task})\n{'='*50}")
    try:
        model = YOLO(weights)
        
        model.train(
            data=data_yaml,
            epochs=300,
            patience=50,
            imgsz=640,
            project="runs/iter1",
            name=name,
            task=task,
            batch=16,
            device="0",  # Explicitly use the primary GPU
            workers=0,   # Disable multiprocessing to prevent WinError 1455 RAM spikes
            save=True,
            exist_ok=True
        )
        
        best_weights = Path(f"runs/iter1/{name}/weights/best.pt")
        if best_weights.exists():
            target_name = models_dir / f"iter1_{name}_best.pt"
            shutil.copy2(best_weights, target_name)
            print(f"Successfully saved {target_name}")
        else:
            print(f"Warning: best.pt not found for {name}")
            
    except Exception as e:
        print(f"Failed to train {name}: {e}")

print("\nAll Iteration 1 experiments completed.")

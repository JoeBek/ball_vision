import os, yaml, json
import io, base64
from PIL import Image
import torch
from ultralytics import YOLO

def init_context(context):
    # Load labels from function.yaml
    with open("/opt/nuclio/function.yaml", 'rb') as f:
        cfg = yaml.safe_load(f)
    labels = {item['id']: item['name'] for item in json.loads(cfg['metadata']['annotations']['spec'])}
    context.user_data.labels = labels


    # TODO set load path
    model_path = "/opt/balls/ball_weights_cheated.pt"
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    
    context.user_data.model = model
    context.logger.info("Model and labels loaded")



def handler(context, event):
    data = event.body
    img_bytes = base64.b64decode(data["image"])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    outputs = context.user_data.model(img)

    results = []
    threshold = float(data.get("threshold", 0.3))
    
    # YOLO returns a list of Results objects
    for result in outputs:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
            scores = result.boxes.conf.cpu().numpy()  # confidence scores
            classes = result.boxes.cls.cpu().numpy()  # class indices
            
            for box, score, cls in zip(boxes, scores, classes):
                if score < threshold:
                    continue
                    
                # Convert class index to label
                label = context.user_data.labels.get(int(cls), f"class_{int(cls)}")
                
                # Convert box coordinates to list
                box_coords = [float(coord) for coord in box]
                
                results.append({
                    "confidence": str(float(score)),
                    "label": label,
                    "points": box_coords,  # [x1, y1, x2, y2]
                    "type": "rectangle"
                })

    return context.Response(body=json.dumps(results),
                            headers={},
                            content_type="application/json",
                            status_code=200)

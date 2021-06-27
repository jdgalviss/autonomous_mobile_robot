import torch
import time
import cv2

colors = [(255,0,0),(0,255,0),(0,0,255), (255,255,255)]
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5x, custom

# Images
img = cv2.imread('dolly.png')[:, :, ::-1]  # or file, PIL, OpenCV, numpy, multiple

# Inference
start = time.time()
for i in range(10):
    results = model(img)
end = time.time()
print('elapsed time: {}'.format(end-start))
result = results.render()[0]
# print(results.imgs.xyxy)
preds = results.xyxy[0].cpu().numpy()
result2 = img.copy()
for p in preds:
    if(p[4] > 0.5):
        class_idx = min(int(p[5]), 3)
        color = colors[class_idx]
        result2 = cv2.rectangle(result2, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), color, 2)
        result2 = cv2.putText(result2, results.names[int(p[5])],  (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 1, cv2.LINE_AA)

result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)     
cv2.imwrite('result.png', result2)

# Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.save() 
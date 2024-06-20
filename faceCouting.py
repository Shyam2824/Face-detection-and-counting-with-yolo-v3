import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


def process_frame(frame):
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
         for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    face_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            if label == "person":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_count += 1

    return frame, face_count

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return
    frame, face_count = process_frame(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    face_count_label.config(text=f"Faces: {face_count}")
    lmain.after(10, update_frame)
    
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        cap.release()
        root.destroy()


cap = cv2.VideoCapture(0)


root = tk.Tk()
root.title("Face Detection App")
root.protocol("WM_DELETE_WINDOW", on_closing)

lmain = tk.Label(root)
lmain.pack()

face_count_label = tk.Label(root, text="Faces: 0", font=("Helvetica", 16))
face_count_label.pack()

update_frame()

root.mainloop()
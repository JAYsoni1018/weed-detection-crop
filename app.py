import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


def load_model():
    return YOLO("model/best.pt")


model = load_model()

st.title("🌱 Weed vs Crop Detection")
st.write("Upload an image and detect weeds using YOLO")

uploaded_files = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
IMAGES_PER_ROW = 3

if uploaded_files:
    if st.button("Check Plant"):
        results = []

        # Run YOLO for each image
        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            img_array = np.array(image)

            preds = model(img_array)
            results.append((file.name, img_array, preds))

        # Display results in grid
        for i in range(0, len(results), IMAGES_PER_ROW):
            cols = st.columns(IMAGES_PER_ROW)

            for col, (name, img_array, preds) in zip(cols, results[i:i + IMAGES_PER_ROW]):
                with col:
                    drawn = img_array.copy()
                    weed_count = 0

                    for r in preds:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        classes = r.boxes.cls.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()

                        for box, cls, conf in zip(boxes, classes, confs):
                            x1, y1, x2, y2 = map(int, box)
                            label = "weed" if int(cls) == 1 else "crop"
                            color = (0, 0, 255) if label == "weed" else (0, 255, 0)

                            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                drawn,
                                f"{label} {conf:.2f}",
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )

                            if label == "weed":
                                weed_count += 1

                    st.image(drawn, width=300)
                    st.caption(f"📄 {name}")
                    st.write(f"🌿 Weeds detected: **{weed_count}**")

import faiss  # Import FAISS for similarity search
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

# Initialize FastAPI app
app = FastAPI()

# Configurations
DATABASE_FOLDER = "./database"
DOWNLOADS_FOLDER = os.path.join(os.path.expanduser("~"), "Downloads")
os.makedirs(DOWNLOADS_FOLDER, exist_ok=True)

# Load pre-trained model
base_model = EfficientNetB4(weights="imagenet", include_top=False, pooling="avg")
model_cnn = Model(inputs=base_model.input, outputs=base_model.output)

# Initialize FAISS Index
faiss_index = None
db_features = []
db_image_names = []

# Function to extract features
def extract_features(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.convert_to_tensor(x)
    features = model.predict(x)
    return features.flatten()

# Load database features into FAISS index
def initialize_faiss():
    global faiss_index, db_features, db_image_names

    for img_name in os.listdir(DATABASE_FOLDER):
        img_path = os.path.join(DATABASE_FOLDER, img_name)
        features = extract_features(img_path, model_cnn)
        db_features.append(features)
        db_image_names.append(img_name)

    db_features = np.array(db_features).astype("float32")
    faiss_index = faiss.IndexFlatL2(db_features.shape[1])
    faiss_index.add(db_features)
    print(f"FAISS index initialized with {faiss_index.ntotal} vectors.")

# Function to generate Grad-CAM visualization
def generate_gradcam(image_path, model, last_conv_layer_name="top_conv", pred_index=None):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img, heatmap

# Function to generate PDF report
def generate_pdf_report(query_img_name, query_img_path, retrieved_results, gradcam_path):
    pdf_path = os.path.join(DOWNLOADS_FOLDER, f"{query_img_name}_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    margin_left = 50
    margin_top = height - 50
    img_width = 200
    img_height = 200
    spacing = 30

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_left, margin_top, f"Medical Image Retrieval Report for: {query_img_name}")

    y_position = margin_top - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_left, y_position, "Query Image:")
    c.drawImage(ImageReader(query_img_path), margin_left, y_position - img_height, width=img_width, height=img_height)

    y_position -= img_height + 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_left, y_position, "Retrieved Images and Scores:")

    y_position -= 20
    for i, (img_name, score, gradcam_img_path) in enumerate(retrieved_results):
        c.setFont("Helvetica", 10)
        c.drawString(margin_left, y_position, f"{i + 1}. {img_name} (Score: {score:.4f})")
        c.drawImage(ImageReader(gradcam_img_path), margin_left, y_position - img_height, width=img_width, height=img_height)
        y_position -= img_height + 30
        if y_position < 100:
            c.showPage()
            y_position = margin_top - 50

    c.save()
    return pdf_path

# Function to retrieve similar images using FAISS
def retrieve_images_faiss(query_feat, top_k=5):
    query_feat = np.array([query_feat]).astype("float32")
    distances, indices = faiss_index.search(query_feat, top_k)
    results = [(db_image_names[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results

# Query Image Processing Workflow
@app.post("/upload/")
async def upload_query_image(file: UploadFile = File(...)):
    global faiss_index

    if faiss_index is None:
        initialize_faiss()

    file_location = os.path.join(DOWNLOADS_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    query_features = extract_features(file_location, model_cnn)
    results = retrieve_images_faiss(query_features, top_k=5)

    gradcam_path = os.path.join(DOWNLOADS_FOLDER, f"{file.filename}_gradcam.jpg")
    gradcam_img, _ = generate_gradcam(file_location, model_cnn)
    cv2.imwrite(gradcam_path, gradcam_img)

    retrieved_results = []
    for img_name, score in results:
        gradcam_retrieved_path = os.path.join(DOWNLOADS_FOLDER, f"{img_name}_gradcam.jpg")
        img_path = os.path.join(DATABASE_FOLDER, img_name)
        gradcam_retrieved_img, _ = generate_gradcam(img_path, model_cnn)
        cv2.imwrite(gradcam_retrieved_path, gradcam_retrieved_img)
        retrieved_results.append((img_name, score, gradcam_retrieved_path))

    report_path = generate_pdf_report(file.filename, file_location, retrieved_results, gradcam_path)

    return JSONResponse({"message": "Processing completed!", "report_url": f"/download/{os.path.basename(report_path)}"})

# Download Endpoint
@app.get("/download/{report_name}")
async def download_report(report_name: str):
    report_path = os.path.join(DOWNLOADS_FOLDER, report_name)
    if os.path.exists(report_path):
        return FileResponse(report_path, media_type="application/pdf", filename=report_name)
    return JSONResponse({"message": "Report not found."}, status_code=404)

if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

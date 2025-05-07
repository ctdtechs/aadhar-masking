from flask import Blueprint, request, jsonify
import logging
import os
from datetime import datetime
import base64
import shutil
import random
from application import AadhaarMasking
import paddle
import time
import gc
from waitress import serve
from crypto_util import encrypt_payload, decrypt_payload

api_blueprint = Blueprint('api', __name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
paddle.enable_static()
paddle.set_device('cpu')  # Force CPU

#masking = AadhaarMasking(yolo_model_path="yollo11m_last.pt", layout_model_tag="model_layoutlm", classes=["Aadhar"], classes2=["personal_info"], gpu=False)

LOG_DIR = "logs"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_filename = f"log-{datetime.now().strftime('%Y-%m-%d')}.txt"
log_filepath = os.path.join(LOG_DIR, log_filename)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, encoding="utf-8", mode="a"),
        logging.StreamHandler()
    ]
)

def aadhaar_masking(input_file_path):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        paddle.enable_static()
        paddle.set_device('cpu')  # Force CPU

        masking = AadhaarMasking(yolo_model_path="yollo11m_last.pt", layout_model_tag="model_layoutlm", orientation_inference_dir="text_image_orientation_infer", classes=["Aadhar"], classes2=["personal_info"], gpu=False)
        start_step1 = time.time()
        outputfile, answers, aadhr_data = masking.run(file_path=input_file_path, output_path=OUTPUT_DIR, threshold=0.5, threshold2=0.91)
        end_step1 = time.time()
        processing_time  = end_step1 - start_step1
        print(outputfile)
        logging.error(f"output: {outputfile}")
        logging.error(f"output: {answers}")
        logging.error(f"output: {aadhr_data}")
        with open(outputfile, "rb") as f:
            byte_array = base64.b64encode(f.read()).decode("utf-8")
        
        # Prepare aadhaar_occurrences list safely
        aadhaar_occurrences = []
        if answers:
            for file_name, file_data in answers.items():
                # If the structure is directly under file_name
                if "aadhaar_occurrences" in file_data:
                    for occurrence in file_data.get("aadhaar_occurrences", []):
                        aadhaar_occurrences.append({
                            "FileName": file_name,
                            **occurrence
                        })

                # If the structure is one level deeper (nested)
                else:
                    for sub_file_name, sub_data in file_data.items():
                        for occurrence in sub_data.get("aadhaar_occurrences", []):
                            aadhaar_occurrences.append({
                                "FileName": f"{file_name}/{sub_file_name}",
                                **occurrence
                            })

        # Extract execution status (handle both cases)
        execution_status = {}
        for item in aadhr_data:
            for key, value in item.items():
                if key == 'Execution':
                    execution_status = {
                        "status": value[0],
                        "error": value[1] if len(value) > 1 else None
                    }
                else:
                    execution_values = item.get(key, {}).get('Execution', [])
                    if execution_values:
                        execution_status = {
                            "FileName": key,
                            "status": execution_values[0],
                            "error": execution_values[1] if len(execution_values) > 1 else None
                        }

        # Construct final result
        result = {
            "aadhaar_occurrences": aadhaar_occurrences,
            "filebytearray": byte_array,
            "processing_time": processing_time,
            "output_filepath": outputfile
        }

        # Add execution status if available
        if execution_status:
            result["execution_status"] = execution_status
            
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
            logging.info(f"Deleted input file: {input_file_path}")

        if os.path.exists(outputfile):
            os.remove(outputfile)
            logging.info(f"Deleted output file: {outputfile}")

        return result

    except Exception as e:
        logging.error(f"Aadhaar masking failed: {e}")
        return {"error": f"Error {e}"}

@api_blueprint.route("/upload-file-json", methods=["POST"])
def upload_file_json():
    encrypted_data = request.get_json().get("data")

    if not encrypted_data:
        return jsonify({"error": "No encrypted data received."}), 400

    data = decrypt_payload(encrypted_data)

    required_keys = ["fileName", "documentIndex", "fileBytes"]

    if not all(key in data for key in required_keys):
        return jsonify({"error": "Missing one or more required fields."}), 400

    file_name = data["fileName"]
    file_bytes = data["fileBytes"]

    if not file_bytes:
        return jsonify({"error": "fileBytes is empty."}), 400

    if 1==1: #try:
        decoded_bytes = base64.b64decode(file_bytes)
        input_file_path = os.path.join(INPUT_DIR, file_name)

        with open(input_file_path, "wb") as f:
            f.write(decoded_bytes)

        logging.info(f"File saved to input folder: {file_name}")

        masking_result = aadhaar_masking(input_file_path)
        
        return jsonify({"data": masking_result}), 200
    
    gc.collect()
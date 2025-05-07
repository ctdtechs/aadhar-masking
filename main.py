
from application import AadhaarMasking
import paddle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

paddle.enable_static()
paddle.set_device('cpu')  # Force CPU

#masking = AadhaarMasking(yolo_model_path="yollo11m_last.pt", layout_model_tag="model_layoutlm", classes=["Aadhar"], classes2=[""], gpu=False)
masking = AadhaarMasking(yolo_model_path="yollo11m_last.pt", layout_model_tag="model_layoutlm", orientation_inference_dir="text_image_orientation_infer",classes=["Aadhar"], classes2=["personal_info"], gpu=False)
#--------------------------------------------
import time


start_step1 = time.time()

outputfile, aadhr_data, answers = masking.run(file_path="/data/Aadhaar/Input/69644354_CLAIM-1122585168879.pdf", output_path="/data/Aadhaar/", threshold=0.5, threshold2=0.91)

end_step1 = time.time()

print(f"Step 1 execution time: {end_step1 - start_step1} seconds")
print(f"\nOutput File Name: {outputfile} ,and the found items are: {aadhr_data}, and the answers are {answers}")

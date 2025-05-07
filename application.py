import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
import re
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from qrdet import QRDetector
import torch
from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
import io
import os
import zipfile
import magic
import olefile
import rarfile
import py7zr
import extract_msg
import traceback
from contextlib import contextmanager
import paddle.inference as paddle_infer
import fitz


Image.MAX_IMAGE_PIXELS = None


class FileProcessor:
    def __init__(self):
        self.mime = magic.Magic(mime=True)

    def get_original_dpi(self, file_path):
        """
        Attempt to estimate the original rendering DPI of a PDF file.
        Note: PDFs are vector-based and don't inherently have a single DPI.
        This method infers it from the dimensions and a default rendering.
        """
        try:
            doc = fitz.open(file_path)
            page = doc[0]  # First page
            rect = page.rect  # Dimensions in points (1 point = 1/72 inch)
            width_inches = rect.width / 72
            height_inches = rect.height / 72

            # Render the page to a pixmap with a matrix of (1, 1).
            # This often corresponds to a default rendering resolution.
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            width_px = pix.width
            height_px = pix.height

            # Calculate approximate DPI
            dpi_x = width_px / width_inches
            dpi_y = height_px / height_inches
            dpi = int(round((dpi_x + dpi_y) / 2))

            # Basic sanity checks and adjustments
            if 50 <= dpi <= 400:
                return dpi
            elif dpi < 50:
                return 72  # Default fallback
            else:
                return 300 # Another common default/upper bound

        except Exception as e:
            return 150
        finally:
            if 'doc' in locals() and doc:
                doc.close() # Ensure the document is closed

    def is_image_file(self, file_content):
        """Check if the file content is an image based on magic numbers."""
        mime_type = self.mime.from_buffer(file_content[:1024])
        image_mime_types = {
            'image/bmp', 'image/gif', 'image/jpeg', 'image/png',
            'image/webp', 'image/apng', 'image/avif', 'image/svg+xml',
            'image/tif', 'image/tiff'
        }
        return mime_type in image_mime_types

    def is_pdf_file(self, file_content):
        """Check if the file content is a PDF."""
        return file_content[:4] == b'%PDF'

    def is_archive_file(self, file_content):
        """Check if the file content is a ZIP, RAR, or 7z archive."""
        archive_signatures = {
            b'PK\x03\x04': 'zip',
            b'Rar!': 'rar',
            b'7z\xBC\xAF\x27\x1C': '7z'
        }
        for signature, format in archive_signatures.items():
            if file_content.startswith(signature):
                return format
        return None

    def is_msg_file(self, file_path):
        return olefile.isOleFile(file_path)

    def _open_archive(self, archive_format, file_path):
        if archive_format == 'zip':
            archive = zipfile.ZipFile(file_path, 'r')
            file_list = archive.namelist()
            get_content = lambda name: archive.read(name)
        elif archive_format == 'rar':
            rarfile.UNRAR_TOOL = "UnRAR.exe"
            archive = rarfile.RarFile(file_path, 'r')
            file_list = archive.namelist()
            get_content = lambda name: archive.read(name)
        elif archive_format == '7z':
            archive = py7zr.SevenZipFile(file_path, mode='r')
            file_list = list(archive.readall().items())
            get_content = None  # handled separately in main function
        else:
            return None, [], None

        return archive, file_list, get_content

    def _convert_to_msg(self, file_path):
        images_ = {}
        msg = extract_msg.Message(file_path)

        for i, attachment in enumerate(msg.attachments):
            filename = attachment.longFilename.translate({0: None})
            file_content = attachment.data

            archive_format = self.is_archive_file(file_content)
            if archive_format:
                    images_.update(self.process_compressed_file(archive_format, file_content=file_content, root_name=filename)[filename])

            elif self.is_pdf_file(file_content):
                dpi_file = self.get_original_dpi(io.BytesIO(file_content))
                print(dpi_file)
                pdf_images = convert_from_bytes(file_content, dpi=dpi_file)
                images_[filename] = [img for img in pdf_images]

            elif self.is_image_file(file_content):
                img = Image.open(io.BytesIO(file_content))
                images_[filename] = [img]

        return images_

    def process_compressed_file(self, archive_format, file_path=None, file_content=None, root_name=""):
        if file_content is not None:
            file_path = io.BytesIO(file_content)

        archive_name = root_name.split('/')[-1]

        # Open the appropriate archive
        archive, file_list, get_content = self._open_archive(archive_format, file_path)
        if archive is None:
            return {}

        images_ = {}

        for file_info in file_list:
            if archive_format == '7z':
                file_name, file_obj = file_info
                archive_content = file_obj.read()
                full_path = file_name
            else:
                archive_content = get_content(file_info)
                full_path = file_info

            full_key = os.path.join(archive_name, full_path)

            inner_format = self.is_archive_file(archive_content)
            if inner_format:
                nested_images = self.process_compressed_file(
                    inner_format, file_content=archive_content, root_name=full_key
                )
                images_.update(nested_images[next(iter(nested_images))])
                continue

            if self.is_pdf_file(archive_content):
                dpi_file = self.get_original_dpi(io.BytesIO(file_content))
                print(dpi_file)
                pdf_images = convert_from_bytes(archive_content, dpi=dpi_file)
                images_[full_key.split('/')[-1]] = [img.convert('RGB') for img in pdf_images]

            elif self.is_image_file(archive_content):
                img = Image.open(io.BytesIO(archive_content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images_[full_key.split('/')[-1]] = [img]
            elif self.is_msg_file(io.BytesIO(archive_content)):
                all_images = self._convert_to_msg(io.BytesIO(archive_content))
                images_.update(all_images)


        archive.close()
        return {archive_name: images_}

    def convert_file_to_images(self, file_path):
        images = {}

        with open(file_path, 'rb') as f:
            file_content = f.read(1024)  # Read first 1KB

        archive_format = self.is_archive_file(file_content)

        if archive_format:
            return self.process_compressed_file(archive_format, file_path=file_path, root_name=file_path)

        if self.is_pdf_file(file_content):
            dpi_file = self.get_original_dpi(io.BytesIO(file_content))
            print(dpi_file)
            pdf_images = convert_from_path(file_path, dpi=dpi_file)
            images[file_path] = [img for img in pdf_images]
            return images

        if self.is_msg_file(file_path):
            
            
            images_ = self._convert_to_msg(file_path)
            images[file_path] = images_
            return images

        if self.is_image_file(file_content):
            img = Image.open(file_path)
            images[file_path] = [img]
            return images


class Reshape:

    def save_to_zip(self, data, output_dir):
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        masked_file_path = ""

        for zip_path, files in data.items():
            split_paths = os.path.splitext(zip_path)
            ext = '.zip' if "msg" in split_paths[1] else split_paths[1]
            masked_file_name = os.path.basename(split_paths[0]) + "_masked" + ext
            masked_file_path = os.path.join(output_dir, masked_file_name)

            with zipfile.ZipFile(masked_file_path, 'w') as zipf:
                for filename, images in files.items():
                    if filename.endswith('.pdf'):
                        img_bytes = io.BytesIO()
                        images[0].save(img_bytes, format='PDF', save_all=True, append_images=images[1:])
                        zipf.writestr(filename, img_bytes.getvalue())
                    elif len(images) == 1:
                        if isinstance(images[0], Image.Image):
                            img_bytes = io.BytesIO()
                            images[0].save(img_bytes, format='PNG')
                            zipf.writestr(filename, img_bytes.getvalue())

        return masked_file_path

    def save_to_pdf(self, data, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        masked_file_path = ""

        for file_path, files in data.items():
            if files:
                filename, ext = os.path.splitext(os.path.basename(file_path))
                masked_file_name = filename + "_masked" + ext
                masked_file_path = os.path.join(output_dir, masked_file_name)

                files[0].save(masked_file_path, save_all=True, append_images=files[1:], optimize=True)

        return masked_file_path

    def save_to_image(self, data, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        masked_file_path = ""

        for file_path, image_list in data.items():
            if image_list:
                filename, ext = os.path.splitext(os.path.basename(file_path))
                masked_file_name = filename + "_masked" + ext
                masked_file_path = os.path.join(output_dir, masked_file_name)

                image_list[0].save(masked_file_path)

        return masked_file_path


class OrientationClassifier:
    def __init__(self, model_path, params_path, use_gpu=False):
        self.angle_map = {
            0: 0,
            1: 90,
            2: 180,
            3: 270,
        }

        # Setup Paddle Inference config
        self.config = paddle_infer.Config(model_path, params_path)
        if use_gpu:
            self.config.enable_use_gpu(100, 0)
        else:
            self.config.disable_gpu()

        self.config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle_infer.create_predictor(self.config)

    def preprocess(self, image):
        image = image.resize((224, 224))
        image = np.asarray(image).astype('float32') / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape((1, 1, 3))

        image = (image - mean) / std
        image = image.transpose((2, 0, 1))  # Convert to CHW
        image = np.expand_dims(image, axis=0).astype('float32')

        return image

    def predict(self, image_path):
        input_image = self.preprocess(image_path)

        input_names = self.predictor.get_input_names()
        input_tensor = self.predictor.get_input_handle(input_names[0])
        input_tensor.copy_from_cpu(input_image)

        self.predictor.run()

        output_names = self.predictor.get_output_names()
        output_tensor = self.predictor.get_output_handle(output_names[0])
        output_data = output_tensor.copy_to_cpu()

        pred_idx = int(np.argmax(output_data))
        predicted_angle = self.angle_map[pred_idx]
        return predicted_angle


class OCR:
    def __init__(self, orientation_inference_dir: str, gpu=False):

        self.OCR = PaddleOCR(
            lang="en",
            det_limit_side_len=800,  # Reduced further for faster processing
            use_gpu=gpu,
            det_db_score_mode="fast",
            show_log=False,
            use_angle_cls=True,
            det_db_unclip_ratio=1.6,  # Optimized for speed and accuracy
            det_db_thresh=0.35,  # Fine-tuned threshold for better balance
            det_db_box_thresh=0.6,  # Slightly higher for more confident detections
            max_batch_size=16,  # Increased batch size for better GPU utilization
            #precision="fp16",  # Maintains half-precision for speed
            #det_model_dir='en_PP-OCRv4_det',
            rec_batch_num=8,  # Added for recognition batch optimization
            drop_score=0.5,  # Added to filter low-confidence results
            use_dilation=False,  # Disable dilation for faster processing
            enable_mkldnn=True  # Enable MKL-DNN for CPU optimization if no GPU
        )
        model_path = orientation_inference_dir + "/inference.pdmodel"
        params_path = orientation_inference_dir + "/inference.pdiparams"

        self.classifier = OrientationClassifier(model_path, params_path, use_gpu=False)


    def process_image(self, image: Image, crop_box):

        image_np = np.array(image)
        if image_np is not None or image_np.size != 0:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            ocr_result = self.OCR.ocr(image_np, cls=True)[0]
            if ocr_result:
                words = [x[1][0] for x in ocr_result]
                boxes = np.asarray([x[0] for x in ocr_result])  # (n_boxes, 4, 2)

                x1_crop, y1_crop, x2_crop, y2_crop = crop_box

                # Adjust bounding box coordinates according to the crop position
                for i, box in enumerate(boxes):
                    boxes[i] = box + np.array([x1_crop, y1_crop])

                x1 = boxes[:, :, 0].min(1)
                y1 = boxes[:, :, 1].min(1)
                x2 = boxes[:, :, 0].max(1)
                y2 = boxes[:, :, 1].max(1)

                boxes = np.stack([x1, y1, x2, y2], axis=1)

                angle = self.classifier.predict(image)

                return {"result": ocr_result, "words": words, "boxes": boxes, "angle": [angle]}

            else: return {"result": [], "words": [], "boxes": [], "angle": []}
        return {"result": [], "words": [], "boxes": [], "angle": []}


class CreateMaskedImage:

    def __init__(self):
        self.all_masked_images = []

    def create_bounding_box(self, bbox_data):
        xs = []
        ys = []
        for x, y in bbox_data:
            xs.append(x)
            ys.append(y)

        left = int(min(xs))
        top = int(min(ys))
        right = int(max(xs))
        bottom = int(max(ys))

        return [left, top, right, bottom]

    def draw(self, image, ocr_result: list):

        font_path = Path(cv2.__path__[0]) / "qt/fonts/DejaVuSansCondensed.ttf"
        font = ImageFont.truetype(str(font_path), size=12)

        left_image = image.convert("RGB")
        left_draw = ImageDraw.Draw(left_image)

        for i, (bbox, (word, confidence)) in enumerate(ocr_result):
            box = self.create_bounding_box(bbox)

            left_draw.rectangle(box, outline="red", width=2)
            left, top, right, bottom = box

            left_draw.text((right + 5, top), text=str(i + 1), fill="orange", font=font)

        self.all_masked_images.append(left_image)

    def save_pdf(self, output_pdf: str):

        # Save all masked images as a single PDF
        if self.all_masked_images:
            self.all_masked_images[0].save(output_pdf, save_all=True, append_images=self.all_masked_images[1:])
            return True
        else: return False


class DetectDocuments:

    def __init__(self, model_path, gpu):
        if gpu: device = "cuda"
        else: device = "cpu"
        self.model = YOLO(model_path).to(device)  # Load the model
        self.model.fuse()

    def detect_objects(self, image):

        image_array = np.array(image)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        results = self.model(image_rgb)[0]
        boxes = results.boxes

        #annotated_image = image_rgb.copy()
        #np.random.seed(42)  # For consistent colors
        #colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

        return boxes, results.names

    def show_results(self, image, confidence_threshold: int, confidence_threshold2: int, classes: list, classes2: list):

        boxes, class_names = self.detect_objects(image)

        class_labels = {"bbox": [], "confidence": [],  "class":[]}

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_name = class_names[int(box.cls[0])]
            print(class_name, confidence)

            if confidence > confidence_threshold and class_name in classes+classes2:
                class_labels["bbox"].append([x1, y1, x2, y2])
                class_labels["confidence"].append(confidence)
                class_labels["class"].append(class_name)

        return class_labels


class PDFExtraction:

    def __init__(self, yolo_model_path, orientation_inference_dir,  gpu=False):

        self.ocr = OCR(orientation_inference_dir=orientation_inference_dir, gpu=gpu)
        self.doc_detection = DetectDocuments(yolo_model_path, gpu)
        self.detector = QRDetector(model_size='l', weights_folder = '.model')
        self.file_processor = FileProcessor()
        self.reshape = Reshape()
        #self.rotate = Orientation_correction()

    def process_page(self, page_data):

        index, preproc_image = page_data
        docs_data = []

        if self.aadhar_scan:
            docs = self.doc_detection.show_results(preproc_image, self.threshold, self.threshold2, self.classes, self.classes2)
            qr_maksing_count = False
            for ind, box in enumerate(docs["bbox"]):
                class_name = docs["class"][ind]

                if class_name in self.classes + self.classes2:
                    crp_img = preproc_image.crop(box).convert("RGB")
                    #2orientation correction code removed
                    image_data = self.ocr.process_image(crp_img, box)
                    docs_data.append([image_data, [class_name, crp_img.size]])
                
                if not qr_maksing_count and class_name in self.classes:
                    # QR Masking ------------------------------
                    image_np = np.array(preproc_image)
                    detections = self.detector.detect(image=image_np, is_bgr=True)

                    for detection in detections:
                        x1, y1, x2, y2 = map(int, detection['bbox_xyxy'])
                        draw = ImageDraw.Draw(preproc_image)
                        draw.rectangle((x1, y1, x2, y2), fill="white")
                    
                    qr_maksing_count = True
                    # -------------------------------------------

        else:
            image_data = self.ocr.process_image(preproc_image)
            docs_data.append(image_data)

        return index, docs_data, preproc_image

    def process_data(self, pages: list):

        all_image_data = {}
        images = []

        # Parallel processing of pages
        with ThreadPoolExecutor() as executor:
            page_data = list(enumerate(pages, start=1))
            futures = [executor.submit(self.process_page, page) for page in page_data]
            results = [future.result() for future in futures]

            for index, docs_data, preproc_image in results:
                all_image_data[index] = docs_data

                # QR Masking code removed ------------------------------

                images.append(preproc_image)

                #if aadhar_scan and docs_data: # want to add the cropped image
                    # Only append images when needed
                    #images.extend([data.get('image') for data in docs_data if data.get('image')])

        return all_image_data, images

    def pdf_to_image(self, file_path, classes, classes2, threshold, threshold2, aadhar_scan=False):
        """
        Input parameter: pdf_file_path - string
        Return: all_image_data - dictionary
          {
            "page_no": [{"ocr_result": [[]], "words": [],"boxes": []}]
          }
        """
        #dpi = self.file_processor.get_original_dpi(file_path)
        #self.convert_params = {'dpi': dpi}
        all_images_data = {}
        self.aadhar_scan = aadhar_scan
        self.classes = classes
        self.threshold = threshold
        self.classes2 = classes2
        self.threshold2 = threshold2

        all_extfiles = self.file_processor.convert_file_to_images(file_path)

        #pages = convert_from_path(pdf_file_path, **self.convert_params)
        for archive_or_file, inner_content in all_extfiles.items():
            if isinstance(inner_content, dict):
                all_images_data[archive_or_file] = {}
                for filename, images_list in inner_content.items():
                    image_data, processed_images = self.process_data(images_list)
                    all_extfiles[archive_or_file][filename] = processed_images
                    all_images_data[archive_or_file][filename] = image_data
            else:
                image_data, processed_images = self.process_data(inner_content)
                all_extfiles[archive_or_file] = processed_images
                all_images_data[archive_or_file] = image_data

        return all_images_data, all_extfiles


class LMVv1Extraction:

    def __init__(self, model_tag):
        self.model_tag = model_tag

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.MODEL = AutoModelForDocumentQuestionAnswering.from_pretrained(self.model_tag).to(self.device).eval()
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.model_tag)

    def extract(self, question, words, boxes):
        encoding = self.TOKENIZER(question, words, boxes, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

        # Move input tensors to the same device as the model
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        # Manually set bbox to the correct dtype (long)
        bbox = torch.tensor(encoding["bbox"], dtype=torch.long).to(self.device)
        encoding["bbox"] = bbox

        output = self.MODEL(**encoding)

        start_logits = output["start_logits"]
        end_logits = output["end_logits"]
        input_ids = encoding["input_ids"]

        if input_ids.ndim > 1:  # More than one dimension
            input_ids = input_ids[0]  # Take the first sequence for example

        start_index = torch.argmax(start_logits).item()
        end_index = torch.argmax(end_logits).item()
        tokens = self.TOKENIZER.convert_ids_to_tokens(input_ids.tolist())

        answer_tokens = tokens[start_index:end_index + 1]
        answer = self.TOKENIZER.convert_tokens_to_string(answer_tokens)

        return answer


class Identification:

    def __init__(self, gpu, classes, classes2, layout_model_tag):
        self.classes = classes
        self.classes2 = classes2

        self.predict_class = LMVv1Extraction(layout_model_tag)

    def find_answer(self, all_images_data):
        answers = []

        question = "what is aadhaar card number"
        #question2 = "what is VID"

        digit_pattern = r'\D'
        aadharVID_digit_pattern = r'\d{11,18}'
        aadhar_pattern = re.compile(r'^\d{11,12}$')
        #VID_pattern = re.compile(r'^(\d{14,18})$')

        def process_file(file_path, images_data):
            file_result = {}
            min_aadhar_length = 11  # Minimum length for Aadhaar
            min_vid_length = 14     # Minimum length for VID

            for page, content in images_data.items():
                found_digits = {}

                if not isinstance(content, dict):
                    content = {page: content}

                for name, entries in content.items():
                    found_numbers = []
                    if entries:
                        for entry in entries:
                            words = entry[0]['words']
                            if not words:
                                continue

                            boxes = np.asarray([x[0] for x in entry[0]["result"]])
                            class_name = entry[1][0]
                            width, height = entry[1][1]

                            x1 = boxes[:, :, 0].min(1) * 1000 / width
                            y1 = boxes[:, :, 1].min(1) * 1000 / height
                            x2 = boxes[:, :, 0].max(1) * 1000 / width
                            y2 = boxes[:, :, 1].max(1) * 1000 / height
                            boxes = np.stack([x1, y1, x2, y2], axis=1)

                            # Handle specific classe
                            if class_name in self.classes2:
                                #matches = [w for w in words if re.search(aadharVID_digit_pattern, w)]
                                matches = [w for w in words if re.search(aadhar_pattern, w)]
                                
                                for m in matches:
                                    cleaned = re.sub(digit_pattern, '', m.replace(" ", ""))
                                    if len(cleaned) >= min_aadhar_length and cleaned not in found_numbers:
                                        found_numbers.append(cleaned)

                                continue

                            # Extract and clean Aadhaar and VID
                            answer_aadhar = re.sub(digit_pattern, '', self.predict_class.extract(question, words, boxes)[1:].replace(" ", ""))
                            #answer_vid = re.sub(digit_pattern, '', self.predict_class.extract(question2, words, boxes).replace(" ", ""))

                            # Validate Aadhaar
                            if answer_aadhar and len(answer_aadhar) >= min_aadhar_length and aadhar_pattern.match(answer_aadhar):
                                if answer_aadhar not in found_numbers:
                                    found_numbers.append(answer_aadhar)
                            else:
                                # Fallback: Search words for valid Aadhaar
                                matches = [w for w in words if aadhar_pattern.match(re.sub(digit_pattern, '', w.replace(" ", "")))]
                                for m in matches:
                                    cleaned = re.sub(digit_pattern, '', m.replace(" ", ""))
                                    if len(cleaned) >= min_aadhar_length and cleaned not in found_numbers:
                                        found_numbers.append(cleaned)

                            # Validate VID
                            #if answer_vid and len(answer_vid) >= min_vid_length and VID_pattern.match(answer_vid):
                            #    if answer_vid not in found_numbers:
                            #        found_numbers.append(answer_vid)
                            #else:
                            #    # Fallback: Search words for valid VID
                            #    matches = [w for w in words if VID_pattern.match(re.sub(digit_pattern, '', w.replace(" ", "")))]
                            #    for m in matches:
                            #        cleaned = re.sub(digit_pattern, '', m.replace(" ", ""))
                            #        if len(cleaned) >= min_vid_length and cleaned not in found_numbers:
                            #            found_numbers.append(cleaned)

                    found_digits[name] = found_numbers
                if page:
                    file_result[page] = found_digits

            return {file_path: file_result} if file_result else None

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, file_path, images_data) for file_path, images_data in all_images_data.items()]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    answers.append(result)

        return answers
    
    def create_filled_box(self, entry, image, ans_list):
        words = entry[0]["words"]
        boxes = entry[0]["boxes"]
        angle = entry[0]["angle"]

        if not isinstance(boxes, list):
            boxes = boxes.tolist()

        for i, word in enumerate(words):
            word = word.replace(" ", "")
            cleaned_word = re.sub(r'\D', '', word)
            if cleaned_word:
                if isinstance(ans_list, dict): ans_list = [data for data in ans_list.values()][0]

                for digit in ans_list:
                    if digit == cleaned_word:
                        draw = ImageDraw.Draw(image)

                        x1, y1, x2, y2 = boxes[i]
                        y1 -= 5  # Expand up
                        y2 += 5  # Expand down
                        x1 -= 5
                        x2 += 5

                        if len(digit) == 12:
                            if angle:
                                if 60 < angle[0] < 120: #90
                                    segment_height = (y2 - y1) / 12  # Segment along Y-axis
                                    for j in range(8):
                                        
                                        seg_y1 = y1 + j * segment_height
                                        seg_y2 = y1 + (j + 1) * segment_height
                                        draw.rectangle([x1, seg_y1, x2, seg_y2], fill="white")

                                elif 240 < angle[0] < 300: #270
                                    segment_height = (y2 - y1) / 12  # Segment along Y-axis
                                    for j in range(8):
                                        seg_y1 = y2 - (j + 1) * segment_height  # Start from bottom
                                        seg_y2 = y2 - j * segment_height
                                        draw.rectangle([x1, seg_y1, x2, seg_y2], fill="white")

                                elif 160 < angle[0] < 200: #180
                                    segment_width = (x2 - x1) / 12 # 12 segment
                                    for j in range(8):
                                        seg_x1 = x2 - (j + 1) * segment_width # Start from bottom
                                        seg_x2 = x2 - j * segment_width
                                        draw.rectangle([seg_x1, y1, seg_x2, y2], fill="white")

                                else:
                                    segment_width = (x2 - x1) / 12 # 12 segment
                                    for j in range(8):
                                        seg_x1 = x1 + j * segment_width  # Small padding
                                        seg_x2 = x1 + (j + 1) * segment_width
                                        draw.rectangle([seg_x1, y1, seg_x2, y2], fill="white")
                        
                            else: draw.rectangle([x1, y1, x2, y2], fill="white")
                        

        return image

    def mask_answer(self, images_data, extfiles, answers):
        new_extfiles = {}
        if answers:
            first_answer = answers[0]
            for top_level_key, top_level_value in first_answer.items():
                if isinstance(top_level_value, dict):  # Type 2: {'--': {'--': ...}}

                    if top_level_key in extfiles and isinstance(extfiles[top_level_key], dict):
                        new_extfiles[top_level_key] = {}
                        for filename, answer in top_level_value.items():
                            if filename in extfiles[top_level_key]:
                                new_extfiles[top_level_key][filename] = list(extfiles[top_level_key][filename])
                                if filename in images_data[top_level_key]:
                                    data = images_data[top_level_key][filename]
                                    images = extfiles[top_level_key][filename]
                                    for indx, ans_list in answer.items():
                                        if indx in data and (indx - 1) < len(images):
                                            for entry in data[indx]:
                                                if (indx - 1) < len(new_extfiles[top_level_key][filename]):
                                                    new_extfiles[top_level_key][filename][indx - 1] = self.create_filled_box(entry, images[indx - 1], ans_list)

                    elif top_level_key in extfiles and not isinstance(extfiles[top_level_key], dict): # Type 1: {'--': {1: ...}}

                        new_extfiles[top_level_key] = list(extfiles[top_level_key])

                        if top_level_key in images_data:
                            data = images_data[top_level_key]
                            images = extfiles[top_level_key]
                            for indx, ans_list in top_level_value.items():
                                if indx in data and (indx - 1) < len(images):
                                    for entry in data[indx]:
                                        if (indx - 1) < len(new_extfiles[top_level_key]):
                                            new_extfiles[top_level_key][indx - 1] = self.create_filled_box(entry, images[indx - 1], ans_list)
        return new_extfiles


class AadhaarMasking:
    def __init__(self, yolo_model_path: str, layout_model_tag:str, orientation_inference_dir:str, classes: list, classes2: list, gpu=False):
        self.yolo_model_path = yolo_model_path
        self.layout_model_tag = layout_model_tag
        self.classes = classes
        self.classes2 = classes2
        self.orientation_inference_dir = orientation_inference_dir
        self.pdf_extractor = None
        self.indetification = None
        self._initialize_resources(gpu)

    def _initialize_resources(self, gpu):
        """Initialize resource-intensive objects."""
        try:
            self.pdf_extractor = PDFExtraction(self.yolo_model_path, self.orientation_inference_dir, gpu=gpu)
            self.indetification = Identification(gpu, self.classes, self.classes2, self.layout_model_tag)
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to initialize resources: {e}")

    def _cleanup(self):
        """Release resources held by external objects."""
        if self.pdf_extractor is not None:
            # Assuming PDFExtraction has a close method; if not, set to None
            if hasattr(self.pdf_extractor, 'close'):
                self.pdf_extractor.close()
            self.pdf_extractor = None
        if self.indetification is not None:
            # Assuming Identification has a close method; if not, set to None
            if hasattr(self.indetification, 'close'):
                self.indetification.close()
            self.indetification = None

    def __del__(self):
        """Destructor to ensure resource cleanup."""
        self._cleanup()

    @contextmanager
    def _temporary_data(self, data):
        """Context manager to ensure temporary data is released."""
        try:
            yield data
        finally:
            data = None  # Help garbage collector release memory

    def scan(self, aadhar_scan: bool, file_path: str, threshold: int, threshold2: int):
        self.aadhar_scan = aadhar_scan
        images_data, all_extfiles = self.pdf_extractor.pdf_to_image(
            file_path=file_path,
            classes=self.classes,
            classes2=self.classes2,
            threshold=threshold,
            threshold2=threshold2,
            aadhar_scan=aadhar_scan
        )
        with self._temporary_data(images_data) as temp_images_data:
            with self._temporary_data(all_extfiles) as temp_extfiles:
                return temp_images_data, temp_extfiles

    def predict(self, images_data, extfiles):
        with self._temporary_data(images_data) as temp_images_data:
            answers = self.indetification.find_answer(temp_images_data)
            with self._temporary_data(answers) as temp_answers:
                new_extfiles = self.indetification.mask_answer(temp_images_data, extfiles, temp_answers)
                return new_extfiles, temp_answers


    def transform_result(self, raw_output):
        result = {}
        try:
            for entry in raw_output:
                for top_key, top_val in entry.items():
                    try:
                        archive_name = top_key
                        result[archive_name] = {}
                        for pdf_filename, pdf_content in top_val.items():
                            aadhaar_occurrences = []
                            if isinstance(pdf_content, dict):
                                for page, values in pdf_content.items():
                                    if values:
                                        aadhaar_occurrences.append({
                                            "page_number": page,
                                            "aadhaar_count": len(values),
                                            "detection_method": "OCR"
                                        })
                            if aadhaar_occurrences:
                                pdf_filename_without_extension = "_".join(os.path.splitext(os.path.basename(pdf_filename))[0].split('_')[:2])
                                result[archive_name][pdf_filename_without_extension] = {"aadhaar_occurrences": aadhaar_occurrences}
                    except:
                        pdf_filename = os.path.basename(top_key)
                        pdf_filename_without_extension = os.path.splitext(pdf_filename)[0]
                        result[pdf_filename] = {"aadhaar_occurrences": []}
                        if isinstance(top_val, dict):
                            for inner_key, inner_val in top_val.items():
                                if isinstance(inner_val, dict):
                                    for page, aadhaar_list in inner_val.items():
                                        if isinstance(aadhaar_list, list):
                                            aadhaars = [a for a in aadhaar_list if a]
                                            if aadhaars:
                                                result[pdf_filename]["aadhaar_occurrences"].append({
                                                    "page_number": str(page),
                                                    "aadhaar_count": len(aadhaars),
                                                    "detection_method": "OCR"
                                                })
                                elif isinstance(inner_val, list):
                                    aadhaars = [a.strip() for a in inner_val if a.strip()]
                                    if aadhaars:
                                        result[pdf_filename]["aadhaar_occurrences"].append({
                                            "page_number": str(inner_key), # Assuming inner_key is the page number
                                            "aadhaar_count": len(aadhaars),
                                            "detection_method": "OCR"
                                        })
                        elif isinstance(top_val, list):
                            aadhaars = [a.strip() for a in top_val if a.strip()]
                            if aadhaars:
                                result[pdf_filename]["aadhaar_occurrences"].append({
                                    "page_number": '1', # Assuming single page if it's a direct list
                                    "aadhaar_count": len(aadhaars),
                                    "detection_method": "OCR"
                                })
            return result

        finally:
            raw_output = None  # Help garbage collector

    def run(self, file_path: str, output_path: str, threshold: int, threshold2: int):
        try:
            images_data, extfiles = self.scan(
                aadhar_scan=True,
                file_path=file_path,
                threshold=threshold,
                threshold2=threshold2
            )
            new_extfiles, answers = self.predict(images_data, extfiles)

            with open(file_path, 'rb') as f:
                file_content = f.read(1024)  # Read first 1KB

            if self.pdf_extractor.file_processor.is_archive_file(file_content) or \
               self.pdf_extractor.file_processor.is_msg_file(file_path):
                outputfile = self.pdf_extractor.reshape.save_to_zip(new_extfiles, output_path)
            elif self.pdf_extractor.file_processor.is_pdf_file(file_content):
                outputfile = self.pdf_extractor.reshape.save_to_pdf(new_extfiles, output_path)
            elif self.pdf_extractor.file_processor.is_image_file(file_content):
                outputfile = self.pdf_extractor.reshape.save_to_image(new_extfiles, output_path)
            else:
                raise ValueError("Unsupported file type")

            aadhr_data = self.transform_result(answers)
            answers[0]["Execution"] = ["Success"]
            return outputfile, aadhr_data, answers

        except Exception as e:
            answers = {}
            answers["Execution"] = [
                "Failed",
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "stack_trace": traceback.format_exc().splitlines()
                }
            ]
            return None, None, answers
        finally:
            images_data = None
            extfiles = None
            new_extfiles = None
            answers = None
            file_content = None


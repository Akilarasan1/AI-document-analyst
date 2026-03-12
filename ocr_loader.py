from langchain.schema import Document
import gc
from rapidocr_onnxruntime import RapidOCR
from threading import Lock
ocr_lock = Lock()

ocr = RapidOCR(
    det_model_path=None,  # Auto-download (~2.5MB)
    rec_model_path=None,  # Auto-download (~10MB)
    cls_model_path=None,  # Set to None to disable
    use_angle_cls=False,  # Disable for less memory
    box_thresh=0.6,       # Higher = fewer detections
    unclip_ratio=1.5,text_score=0.5,use_det=True,
    use_cls=False,        # Saves significant memory
    use_rec=True,device='cpu',         # Force CPU for consistent memory
    print_verbose=False   # Disable logs in production
)


def load_image_document(image_file):
    try:
        img_info = []
        with ocr_lock:
            result, _ = ocr(image_file)
            for item in result:
                text = item[1] 
                img_info.append(text)

        print(f"Extracted text from {image_file} using Rapid OCR: {img_info}")
        doc = Document(page_content="\n".join(img_info),metadata={"source": image_file})    
        gc.collect()
        return [doc]

    except Exception as e:
        print(f"Error while procesing Rapid ocr file :: {e}")
        return []

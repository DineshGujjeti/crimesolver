import cv2, torch, numpy as np, io, os, hashlib, json
from PIL import Image, ExifTags
from torchvision import transforms, models
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

def _load_class_names():
    p = os.path.join(MODELS_DIR, "class_names.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"classifier": ["normal","violence"]}

CRIME_CLASSES     = _load_class_names().get("classifier", ["normal","violence"])
DANGEROUS_OBJECTS = {"gun","pistol","rifle","weapon","knife","blood","fire"}

COCO_MAP = {
    0:"person", 1:"bicycle", 2:"car", 3:"motorcycle", 5:"bus", 7:"truck",
    14:"bird", 15:"cat", 16:"dog", 24:"backpack", 25:"umbrella",
    26:"handbag", 28:"suitcase", 39:"bottle", 41:"cup",
    43:"knife", 44:"spoon", 45:"bowl", 56:"chair", 57:"couch",
    60:"table", 62:"tv", 63:"laptop", 67:"cell phone",
    73:"book", 76:"scissors", 77:"teddy bear",
}
COCO_REMAP = {"scissors":"knife", "knife":"knife"}

COLORS = {
    "gun":(0,0,255),"pistol":(0,0,255),"rifle":(0,0,200),"weapon":(0,0,180),
    "knife":(0,80,255),"blood":(60,0,180),"fire":(0,120,255),
    "person":(0,210,80),"car":(220,180,0),"motorcycle":(200,140,0),
    "bicycle":(180,120,0),"truck":(160,100,0),"bus":(140,80,0),
    "bottle":(140,200,100),"backpack":(100,180,200),"cell phone":(180,100,220),
}
DEF_COLOR = (160,160,160)

VAL_TF = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class CrimeAnalyzer:

    def __init__(self):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_custom = None
        self.yolo_coco   = None
        self.classifier  = None
        self.models_ready= False
        self._load()

    def _load(self):
        try:
            p = os.path.join(MODELS_DIR, "best.pt")
            if os.path.exists(p):
                self.yolo_custom = YOLO(p)
                print(f"✅ Custom YOLO loaded from {p}")
            else:
                print(f"⚠️ best.pt not found at {p}")

            print("⬇️ Loading YOLOv8s pretrained (COCO)...")
            self.yolo_coco = YOLO("yolov8s.pt")
            print("✅ YOLOv8s COCO ready")

            cp = os.path.join(MODELS_DIR, "crime_classifier_best.pt")
            if os.path.exists(cp):
                m = models.efficientnet_b0()
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CRIME_CLASSES))
                ckpt = torch.load(cp, map_location=self.device, weights_only=False)
                m.load_state_dict(ckpt["model_state_dict"])
                self.classifier = m.to(self.device).eval()
                print("✅ Classifier loaded")
            else:
                print(f"⚠️ Classifier not found at {cp}")

            self.models_ready = True
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            self.models_ready = False

    def validate_image(self, img_bytes: bytes, filename: str) -> dict:
        issues, warnings = [], []

        if not any(img_bytes[:len(s)]==s for s in
                   [b"\xff\xd8\xff",b"\x89PNG",b"GIF8",b"RIFF",b"BM"]):
            issues.append("File signature mismatch")

        try:
            pil = Image.open(io.BytesIO(img_bytes))
            pil.verify()
            pil = Image.open(io.BytesIO(img_bytes))
            w, h = pil.size
            mode = pil.mode
        except Exception as e:
            return {"is_valid":False,"is_authentic":False,
                    "ela_score":0,"issues":[str(e)],"warnings":[],"metadata":{},"image_info":{}}

        ela = self._ela(pil)
        auth = ela < 15.0
        meta = self._exif(pil, filename, img_bytes)

        if not auth:
            warnings.append(f"Possible manipulation (ELA {ela:.1f})")

        if w < 32 or h < 32:
            issues.append("Resolution too low")

        return {
            "is_valid":len(issues)==0,
            "is_authentic":auth,
            "ela_score":round(ela,2),
            "ela_threshold":15.0,
            "issues":issues,
            "warnings":warnings,
            "metadata":meta,
            "image_info":{
                "width":w,
                "height":h,
                "mode":mode,
                "file_size":f"{len(img_bytes)/1024:.1f} KB",
                "md5_hash": hashlib.md5(img_bytes).hexdigest()
            }
        }

    def _ela(self, pil, quality=90):
        try:
            buf = io.BytesIO()
            rgb = pil.convert("RGB")
            rgb.save(buf,"JPEG",quality=quality)
            buf.seek(0)
            comp = Image.open(buf).convert("RGB")

            diff = np.abs(np.array(rgb,dtype=np.float32) - np.array(comp,dtype=np.float32)).flatten()
            diff.sort()

            return float(np.mean(diff[int(len(diff)*0.9):]))
        except:
            return 0.0

    def _exif(self, pil, filename, img_bytes):
        meta = {"filename":filename,"file_size":f"{len(img_bytes)/1024:.1f} KB","format":pil.format}

        try:
            ex = pil._getexif()
            if ex:
                for tid,val in ex.items():
                    tag = ExifTags.TAGS.get(tid,tid)
                    if tag in ["DateTime","Make","Model","Software","DateTimeOriginal"]:
                        meta[tag] = str(val)
        except:
            pass

        return meta

    def analyze(self, img_path:str, img_bytes:bytes, filename:str) -> dict:

        validation  = self.validate_image(img_bytes, filename)
        pil         = Image.open(img_path).convert("RGB")

        detections  = self._detect(img_path)
        cls_result  = self._classify(pil)

        description = self._describe(cls_result, detections)
        threat      = self._threat(cls_result, detections)

        ann_path    = self._annotate(img_path, detections)

        return {
            "validation":validation,
            "classification":cls_result,
            "detections":detections,
            "description":description,
            "threat_level":threat,
            "annotated_image":ann_path,
            "total_objects":len(detections),
            "dangerous_objects":[d for d in detections if d["object"] in DANGEROUS_OBJECTS]
        }

    def _detect(self, img_path:str) -> list:

        dets, boxes = [], []

        if self.yolo_coco:
            try:
                res = self.yolo_coco(img_path, conf=0.25, iou=0.45, verbose=False)[0]

                for box in res.boxes:

                    cid  = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    if cid not in COCO_MAP:
                        continue

                    raw  = COCO_MAP[cid]
                    name = COCO_REMAP.get(raw, raw)

                    dets.append({
                        "object":name,
                        "raw_label":raw,
                        "confidence":round(conf,3),
                        "source":"coco",
                        "box":{
                            "x1":int(xyxy[0]),
                            "y1":int(xyxy[1]),
                            "x2":int(xyxy[2]),
                            "y2":int(xyxy[3])
                        },
                        "is_dangerous":name in DANGEROUS_OBJECTS
                    })

                    boxes.append(xyxy)

            except Exception as e:
                print(f"COCO error: {e}")

        if self.yolo_custom:
            try:
                res = self.yolo_custom(img_path, conf=0.50, iou=0.45, verbose=False)[0]

                for box in res.boxes:

                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    if self._overlap(xyxy, boxes, 0.3):
                        continue

                    dets.append({
                        "object":"gun",
                        "raw_label":"gun",
                        "confidence":round(conf,3),
                        "source":"custom",
                        "box":{
                            "x1":int(xyxy[0]),
                            "y1":int(xyxy[1]),
                            "x2":int(xyxy[2]),
                            "y2":int(xyxy[3])
                        },
                        "is_dangerous":True
                    })

                    boxes.append(xyxy)

            except Exception as e:
                print(f"Custom YOLO error: {e}")

        dets.sort(key=lambda x: x["confidence"], reverse=True)

        print(f" → {len(dets)} objects: {[d['object'] for d in dets]}")

        return dets

    def _overlap(self, b, existing, thresh=0.3):

        x1,y1,x2,y2 = b

        for e in existing:

            ex1,ey1,ex2,ey2 = e

            ix1 = max(x1,ex1)
            iy1 = max(y1,ey1)
            ix2 = min(x2,ex2)
            iy2 = min(y2,ey2)

            if ix2<=ix1 or iy2<=iy1:
                continue

            inter = (ix2-ix1)*(iy2-iy1)
            union = (x2-x1)*(y2-y1)+(ex2-ex1)*(ey2-ey1)-inter

            if union>0 and inter/union>thresh:
                return True

        return False


    def _predict_crime_type(self, cls, dets):

        objects = [d["object"] for d in dets]
        scene = cls.get("scene_type","unknown")

        if "gun" in objects or "pistol" in objects or "rifle" in objects:
            return "Armed Robbery"

        if "knife" in objects and "person" in objects:
            return "Assault with Weapon"

        if objects.count("person") >= 2 and scene == "violence":
            return "Physical Assault"

        if "car" in objects or "truck" in objects or "motorcycle" in objects:
            if scene == "violence":
                return "Road Rage / Vehicle Assault"
            return "Traffic Accident"

        if "backpack" in objects and "person" not in objects:
            return "Suspicious Object"

        if "fire" in objects:
            return "Arson / Fire Hazard"

        if scene == "violence":
            return "Violent Activity"

        return "Normal Activity"


    def _classify(self, pil):

        if not self.classifier:
            return {"scene_type":"unknown","confidence":0.0,"probabilities":{}}

        try:
            t = VAL_TF(pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                p = torch.softmax(self.classifier(t), dim=1)[0]

            idx = p.argmax().item()

            return {
                "scene_type":CRIME_CLASSES[idx],
                "confidence":round(float(p[idx]),3),
                "probabilities":{c:round(float(p[i]),3) for i,c in enumerate(CRIME_CLASSES)}
            }

        except Exception as e:
            print(f"Classifier error: {e}")
            return {"scene_type":"unknown","confidence":0.0,"probabilities":{}}


    def _describe(self, cls, dets):

        scene,conf = cls.get("scene_type","unknown"), cls.get("confidence",0)

        crime_type = self._predict_crime_type(cls, dets)

        lines = []
        lines.append(f"🔎 Predicted Crime Type: {crime_type}")

        if scene=="violence":
            lines.append(f"⚠️ VIOLENT scene classified with {conf*100:.1f}% confidence.")
        elif scene=="normal":
            lines.append(f"✅ NON-VIOLENT scene with {conf*100:.1f}% confidence.")
        else:
            lines.append("Scene classification inconclusive.")

        counts = {}

        for d in dets:
            counts[d["object"]] = counts.get(d["object"],0)+1

        if counts:
            lines.append("Objects: " + ", ".join(f"{v}× {k}" for k,v in counts.items()) + ".")

        dangerous = list({d["object"] for d in dets if d.get("is_dangerous")})

        if dangerous:
            lines.append(f"🚨 DANGEROUS items: {', '.join(dangerous)}.")

        persons = sum(1 for d in dets if d["object"]=="person")

        if persons==1:
            lines.append("1 individual visible.")
        elif persons>1:
            lines.append(f"{persons} individuals visible.")

        lines.append("Evidence logged for review.")

        return " ".join(lines)


    def _threat(self, cls, dets):

        score = 0

        if cls.get("scene_type")=="violence":
            score += 40 * cls.get("confidence",0)

        W = {"gun":35,"pistol":35,"rifle":35,"weapon":30,"knife":25,"blood":20,"fire":20}

        for d in dets:
            score += W.get(d["object"],3)*d["confidence"]

        score = min(100, score)

        if score>=70:
            level,color="CRITICAL","red"
        elif score>=40:
            level,color="HIGH","orange"
        elif score>=20:
            level,color="MODERATE","yellow"
        else:
            level,color="LOW","green"

        return {"level":level,"score":round(score,1),"color":color}


    def _annotate(self, img_path:str, dets:list):

        try:
            img = cv2.imread(img_path)

            if img is None:
                return img_path

            for d in dets:

                b = d["box"]
                x1,y1,x2,y2 = b["x1"],b["y1"],b["x2"],b["y2"]

                color = COLORS.get(d["object"], DEF_COLOR)

                label = f'{d["object"].upper()} {d["confidence"]*100:.0f}%'

                cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

                font,fs,ft = cv2.FONT_HERSHEY_SIMPLEX,0.52,1

                (tw,th),bl = cv2.getTextSize(label,font,fs,ft)

                pad=4

                lx1,ly1,lx2,ly2 = x1, max(0,y1-th-pad*2-bl), x1+tw+pad*2, y1

                overlay = img.copy()

                cv2.rectangle(overlay,(lx1,ly1),(lx2,ly2),color,-1)

                cv2.addWeighted(overlay,0.8,img,0.2,0,img)

                cv2.putText(img,label,(lx1+pad,ly2-bl-2),font,fs,(255,255,255),ft,cv2.LINE_AA)

            base,ext = os.path.splitext(img_path)

            out = f"{base}_annotated{ext}"

            cv2.imwrite(out,img)

            return out

        except Exception as e:
            print(f"Annotation error: {e}")
            return img_path   
        
# ───────────── FORENSIC REPORT GENERATOR ─────────────

def _generate_report(self, cls, dets, threat):

    crime_type = self._predict_crime_type(cls, dets)

    objects = [d["object"] for d in dets]

    evidence = ", ".join(set(objects)) if objects else "No visible evidence"

    persons = objects.count("person")

    if persons > 0:
        people_text = f"{persons} individual(s) detected"
    else:
        people_text = "No persons detected"

    recommendation = "Monitor situation"

    if threat["level"] == "CRITICAL":
        recommendation = "Immediate law enforcement response required."

    elif threat["level"] == "HIGH":
        recommendation = "Security intervention recommended."

    elif threat["level"] == "MODERATE":
        recommendation = "Situation should be monitored."

    report = f"""
CRIME SCENE REPORT

Possible Crime Type: {crime_type}

Threat Level: {threat['level']} (Score: {threat['score']})

Evidence Detected:
{evidence}

People Detected:
{people_text}

Recommended Action:
{recommendation}

AI Forensic System Analysis Complete.
"""

    return report.strip()


# ───────────── MODIFY analyze() FUNCTION ─────────────

def analyze(self, img_path:str, img_bytes:bytes, filename:str) -> dict:

    validation  = self.validate_image(img_bytes, filename)
    pil         = Image.open(img_path).convert("RGB")

    detections  = self._detect(img_path)
    cls_result  = self._classify(pil)

    description = self._describe(cls_result, detections)

    threat      = self._threat(cls_result, detections)

    # NEW: Generate forensic report
    report = self._generate_report(cls_result, detections, threat)

    ann_path    = self._annotate(img_path, detections)

    return {
        "validation":validation,
        "classification":cls_result,
        "detections":detections,
        "description":description,
        "forensic_report": report,
        "threat_level":threat,
        "annotated_image":ann_path,
        "total_objects":len(detections),
        "dangerous_objects":[d for d in detections if d["object"] in DANGEROUS_OBJECTS],
    }
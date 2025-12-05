# ⚽ Soccer Analytics Project

A Full-Stack AI Pipeline for Football Analytics using StatsBomb, YOLOv8 & Streamlit

This project combines event data, machine learning, and computer vision to create an end-to-end analytics system capable of predicting match outcomes, evaluating player scoring probability, and analyzing match footage using an AI-powered video pipeline.

---

## 🌟 Key Features

### 🏆 Model 1 — Match Outcome Predictor

Predicts **Home Win / Draw / Away Win** using advanced rolling features:
- Rolling xG, possession %, passes (last 5 matches)
- Team form indicators

**Models used:** Logistic Regression, Random Forest  
**Training Dataset:** 6 seasons of La Liga (~2,200 matches)

### ⚽ Model 2 — Player Scoring Probability

Predicts the likelihood of a player scoring in the next match.

**Features:**
- xG per 90
- Shots per 90
- Minutes played
- Team attacking intensity

**Model:** Class-Weighted Logistic Regression

### 🎥 Model 3 — Computer Vision (VisionPro)

A YOLOv8-based video intelligence system:
- Player & ball detection
- Team classification via jersey color clustering
- Possession tracking based on proximity algorithm
- Output video with real-time overlays

**Technologies:** YOLOv8, OpenCV, Numpy

### 📊 Interactive Streamlit Dashboard

Includes:
- Match outcome predictions
- Player scoring predictions
- Video processing with YOLOv8
- Embedded visualization

**Run using:**
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
soccer-analytics-project/
├── app.py                       # Streamlit dashboard
├── config.py                    # System configuration
├── INSTRUCTIONS.pdf             # Annotation guidelines (HW3)
├── cvat_labels.json             # CVAT → YOLO label config
├── potato_project/              # Potato text annotation workspace
├── data/
│   ├── raw/                     # StatsBomb JSON event files
│   ├── video_raw/               # Raw MP4 match videos
│   ├── images_for_annotation/   # Extracted frames (HW3)
│   └── processed/               # Clean CSVs & output MP4s
├── models/                      # Trained ML models (.pkl)
├── src/
│   ├── computer_vision/         # YOLOv8 detection, TeamID
│   ├── data/                    # Data ingestion + cleaning
│   ├── features/                # Feature engineering
│   └── models/                  # ML training pipelines
├── requirements.txt
└── README.md
```

---

## 📝 Data Annotation (ARI 510 — HW3)

The project includes a formal annotation workflow for both event text labeling and vision labeling.

### Dataset Contents
- Event logs (text commentary)
- Video frames for player/ball annotation

📁 **Dataset link:**  
[https://drive.google.com/drive/folders/18_k6F3K_nRStpZuSVitwCNxoTgqh2Okl?usp=sharing](https://drive.google.com/drive/folders/18_k6F3K_nRStpZuSVitwCNxoTgqh2Okl?usp=sharing)

### 🔧 Annotation Tools

#### 1. Potato — Event Text Annotation
Used for labeling event descriptions.  
**Config:** `potato_project/configs/config.yaml`

#### 2. CVAT — Player & Ball Annotation
Used to draw bounding boxes.  
**Export format:** YOLO 1.1  
Follows rules described in `INSTRUCTIONS.pdf`

### 📘 Annotation Guidelines (Summary)

**Event Tagging Rules:**
- Label based on intent, not surface wording
- Example: A cross that enters the net → Shot

**Vision Labeling Rules:**
- Bounding boxes must be tight
- Skip objects more than 50% occluded
- Maintain consistent color labeling for teams

**Full guidelines:** `INSTRUCTIONS.pdf`

---

## 💻 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd soccer-analytics-project
```

### 2️⃣ Create & activate virtual environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the dashboard
```bash
streamlit run app.py
```

---

## 📊 Data Sources

| Source | Description |
|--------|-------------|
| StatsBomb Open Data | High-quality football event-level dataset |
| Custom Video Dataset | Raw match recordings used for CV model |

---

## 🧠 Model Summary

| Component | Description |
|-----------|-------------|
| Feature Engineering | Rolling windows, team form metrics |
| Match Model | Logistic Regression, Random Forest |
| Player Model | Class-Weighted Logistic Regression |
| Vision Model | YOLOv8 trained on annotated CVAT frames |
| Dashboard | Streamlit-based web UI |
| Outputs | Predictions, probabilities, annotated videos |

---

## 📈 Results Summary

### ML Results
- Evaluated with Accuracy, F1-Score, ROC-AUC
- Stable performance across 6-season La Liga dataset

### CV Results
- Tested on 300+ annotated frames
- Consistent detection of players, ball, and jersey colors

**Why results are reliable:**
- Clean StatsBomb data
- Strong handcrafted features
- Strict annotation guidelines
- State-of-the-art YOLOv8 detection model

---

## ⚠️ Limitations

- Jersey colors can confuse team classification
- Occlusion lowers ball detection accuracy
- Limited custom training data reduces generalization
- Tactical/formation context not captured fully in numerical data

---

## 🔮 Future Work

- Add Expected Threat (xT) modeling
- Expand YOLO training dataset
- Use Optical Flow for better ball tracking
- Build a pass network visualization module
- Cloud deployment for real-time match analytics

---

## 🤝 Contributing

Contributions are welcome!  
Fork → Branch → Pull Request.

---

## 📜 License

This project is released under the MIT License.

---

## ⭐ Acknowledgements

- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- University of Michigan-Flint (ARI 510)
- Open-source sports analytics community
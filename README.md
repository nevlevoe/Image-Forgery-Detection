# HPE CPP project by Aneesh K P , Gargi Bharadwaj , Mukund Raghavan Sadavarthi , Preethi Narasimhan , Rishab Kumar
# 🕵️ Image Forgery Detection

A lightweight and effective image forgery detection tool using multiple forgery methods. Includes pretrained models and a Streamlit-based interactive interface.

---

## 📦 Requirements

Install the required Python packages:

```bash
pip install -r versionstxt.txt
```

> ✅ Recommended Python version: **3.10**

---

## 📥 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/nevlevoe/Image-Forgery-Detection.git
cd Image-Forgery-Detection
```

---

### 2️⃣ Download Pretrained Model Files

The following model files are **not included** in the repository due to size limits:

- `casiamvssnet.pt`
- `defactomvssnet.pt`
- `best_combined_resnet_model.pth`
- `model.weights.pth`
- `modelfinal.weights.pth`

📂 Download and extract all of them from this [Google Drive link](https://drive.google.com/file/d/1CygKYYN82a3kPRgSQVfmmNoE_Kwh9QtI/view?usp=sharing) and place them:

- In the **same directory** as `app_traditionalforg.py`, **or**
- In the **root folder** of the cloned repo

---

### 3️⃣ Run the App via Streamlit

Once dependencies are installed and models are in place, run:

```bash
streamlit run app_traditionalforg.py
```

✅ The app will launch in your browser. Upload an image and see forgery detection results powered by pretrained models.
📂 Download and extract some test pictures from [Google Drive link](https://drive.google.com/file/d/1_kVoFBsQS2n6Di6FIdVxmiF8vt-sDFcW/view?usp=sharing) and use them to upload and test the model:

---

## ✅ Summary

| Step                      | Command / Action                                     |
|---------------------------|------------------------------------------------------|
| Clone Repo                | `git clone ...` → `cd Image-Forgery-Detection`       |
| Install Dependencies      | `pip install -r versionstxt.txt`                    |
| Download Model Files      | From Google Drive → place next to `.py` app file     |
| Run Streamlit App         | `streamlit run app_traditionalforg.py`              |


✅ #*Project Overview Goal:*
- Develop a system to:
      - → Classify solar panel images by condition (e.g., Dusty, Clean)
      - → Deploy results via an interactive Streamlit app for real-time inspection and maintenance recommendations

🔧 ##*Key Components:*

**1. Image Classification (Deep Learning – CNN):**
   ***Goal:*** Predict the condition of solar panels
   ***Target:*** One of six classes (Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, Snow-Covered)
   ***Model Suggestions:*** ResNet, MobileNet, VGG, EfficientNet
   ***Aim:*** Automate classification of solar panels to prioritize maintenance


🚀 ##*Approach Summary*
1. **Data Cleaning & Preprocessing**
   * Resize images to 224x224
   * Normalize pixel values
   * Annotate images for object detection using LabelImg
   * Perform image augmentation to balance classes

2. **📊 Exploratory Data Analysis (EDA)**
   * Visual inspection of defect samples
   * Analyze class distribution
   * Identify patterns in defect frequency
   * Compare occurrences across conditions (e.g., more snow in winter months)

3. **Model Training**
   * Classification: CNN (ResNet, MobileNet, etc.)

4. **🧪 Model Evaluation**
   * Classification Metrics: Accuracy, Precision, Recall, F1-Score

5. **💻 Deployment (Streamlit App)**
-  Users upload images to get:
   - → Panel condition classification

#*Final Output:*
1. **Classification:** Accuracy, Precision, Recall, F1-Score
2. **Streamlit Application:** Real-time image analysis and visualization



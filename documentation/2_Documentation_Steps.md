# Step1: Load DataSet:
- Importing Librarires.
- Loading the Datasets and saving in the dataset folder
- Count of Images:
    - ○ Bird-Drop: 191 images
    - ○ Clean: 193 images
    - ○ Dusty: 190 images
    - ○ Electrical-Damage: 103 images
    - ○ Physical-Damage: 69 images
    - ○ Snow-Covered: 123 images
    - ○ Total images: 869
# Step2: Splitting train and test images:
    - we got colored image, and we got 3D values.

    - In tensor flow, we have an option to download the images from directory.

    - we are splitting the train images and test images.
    - Verifying Number of training images & testing images
# Step3: Model Training
    - CNN for Solar Panel Defect Classification - Sequential model
    - Loss & Optimizer:
           - Loss: Categorical Cross-Entropy
           - Optimizer: Adam / SGD
    - Accuracy vs loss for Sequential Model
# Step 4: Model Evaluation
   - Saving the Model

   - 📈 Classification Metrics:
      - Accuracy
      - Precision, Recall, F1-Score (Per class and macro/micro avg)
      - Confusion Matrix
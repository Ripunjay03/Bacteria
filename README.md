# Bacteria Image Classifier

This project is a machine learning application for classifying microscopic images of bacteria using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The application provides a web interface using Streamlit for easy image upload and prediction.

## Features

- Classify bacteria images into 9 categories: Amoeba, Euglena, Hydra, Paramecium, Rod Bacteria, Spherical Bacteria, Spiral Bacteria, Yeast
- Web-based interface using Streamlit
- Pre-trained CNN model for accurate predictions
- Easy-to-use upload functionality

## Dataset

The dataset consists of microscopic images of various bacteria types, organized into folders by class. The model was trained on this dataset using data augmentation techniques.

## Requirements

- Python 3.8+
- Streamlit
- TensorFlow 2.x
- NumPy
- Pillow (PIL)
- Pandas
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ripunjay03/Bacteria.git
   cd Bacteria
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and go to the provided URL (usually http://localhost:8501)

3. Upload a bacteria image (PNG, JPG, JPEG) using the file uploader

4. The app will display the uploaded image and predict the bacteria type

## Running in GitHub Codespace

This project can be easily run in GitHub Codespace:

1. Go to the repository on GitHub: https://github.com/Ripunjay03/Bacteria
2. Click on the "Code" button
3. Select "Open with Codespace"
4. Once the Codespace is ready, open the terminal and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. Click on the "Ports" tab and open port 8501 to access the Streamlit app

## Model Training

To retrain the model or train on new data:

1. Ensure your dataset is organized in folders by class
2. Run the training script:
   ```bash
   python bacteria_classifier.py
   ```

The script will:
- Load and preprocess the images
- Build and train a CNN model
- Save the trained model as `bacteria_model.h5`
- Save class indices as `class_indices.npy`

## Project Structure

```
Bacteria/
├── app.py                    # Streamlit web application
├── bacteria_classifier.py    # Model training script
├── requirements.txt          # Python dependencies
├── bacteria_model.h5         # Trained model (generated)
├── class_indices.npy         # Class indices (generated)
├── README.md                 # This file
└── [Dataset folders]/        # Image folders for each bacteria type
    ├── Amoeba/
    ├── Euglena/
    ├── Hydra/
    ├── Paramecium/
    ├── Rod Bacteria/
    ├── Spherical Bacteria/
    ├── Spiral Bacteria/
    └── Yeast/
```

## Contributing

Feel free to contribute to this project by:
- Improving the model architecture
- Adding more bacteria types
- Enhancing the web interface
- Optimizing performance

## License

This project is open-source and available under the MIT License.

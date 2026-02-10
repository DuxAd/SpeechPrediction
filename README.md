# Audio Command Recognition with CNN

This project implements a Deep Learning model to classify audio commands (e.g., "up", "down", "left", "right", "stop", "yes", "no") using the **Speech Commands** dataset from tensorflow. The pipeline transforms raw audio waveforms into **Mel-Spectrograms** and processes them through a custom CNN architecture.

## Features
- **Data Balancing:** Handles class imbalance (specifically for the `_unknown_` class) using custom filtering.
- **Signal Processing:** Implements Short-Time Fourier Transform (STFT) and Mel-frequency scaling using TensorFlow's signal processing tools.
- **Data Augmentation:** Includes random noise injection to improve model robustness.
- **Dual-Path Convolution:** The model uses parallel 1D-like convolutions (Time-wise and Frequency-wise) to capture distinct audio features before merging them.

## Model Architecture
The core of the model (`AudioCNNModel`) features:
1. **Input:** Mel-spectrograms of shape `(99, 64, 1)`.
2. **Feature Extraction:** - A parallel branch for **Temporal** features (Kernel 1x5).
   - A parallel branch for **Frequency** features (Kernel 5x1).
3. **Deep Layers:** Multiple Conv2D layers with BatchNormalization and MaxPooling.
4. **Regularization:** Dropout (0.3) and Early Stopping during training.

## Results
The model achieves high accuracy on the validation set. Below are the training metrics and performance analysis:

### Training Performance
| Accuracy | Loss |
| :---: | :---: |
| <img width="640" height="480" alt="Accuracy" src="https://github.com/user-attachments/assets/4ce7fc07-c578-4f9c-81a3-e7f74b949028" /> | <img width="640" height="480" alt="Losses" src="https://github.com/user-attachments/assets/18420804-ef50-4017-8c5a-47f071b77a71" /> |

*The model reaches >95% validation accuracy. Note: Early stopping is implemented to prevent overfitting.*

### Confusion Matrix
The matrix shows strong performance across most classes, with a high concentration on the diagonal.
<img width="754" height="567" alt="Confusion" src="https://github.com/user-attachments/assets/ed84d334-5bbf-4cdd-a10a-2d99144f185b" />

### Visualization
Sample Mel-spectrograms used as input for the CNN:
<img width="600" height="400" alt="Mel_Spectrogram" src="https://github.com/user-attachments/assets/f1c73657-411a-46d3-af33-720ed4567ccb" />

### Performance metrics 
The model was evaluated on 21,299 samples. Below are the detailed metrics for each class (Labels 0-11):

| Class | Precision | Recall | F1-score | Support |
| :--- | :---: | ---: | ---: | ---: |
|           0 |      0.89 |     0.94 |     0.92 |      795 |
|           1 |      0.90 |     0.90 |     0.90 |      765 |
|           2 |      0.94 |     0.95 |     0.95 |      788 |
|           3 |      0.91 |     0.95 |     0.93 |      761 |
|           4 |      0.95 |     0.91 |     0.93 |      725 |
|           5 |      0.90 |     0.94 |     0.92 |      786 |
|           6 |      0.86 |     0.98 |     0.92 |      756 |
|           7 |      0.96 |     0.96 |     0.96 |      755 |
|           8 |      0.88 |     0.95 |     0.92 |      699 |
|           9 |      0.98 |     0.96 |     0.97 |      860 |
|          10 |      0.93 |     0.98 |     0.95 |      135 |
|          11 |      0.98 |     0.97 |     0.97 |    13474 |
| Overall Accuracy | | |			0.96 |	 21,299|

### Short Interpretation
-  High General Accuracy: With a global accuracy of 96%, the model demonstrates a strong ability to distinguish between different voice commands.

-  Class 11 Performance: The _unknown_ class (Label 11), which represents the majority of the data, is handled exceptionally well with a 0.98 precision. This indicates that the model is very reliable at not "hallucinating" commands when the audio is just background noise or irrelevant speech.

-  Reliability (Recall): The Macro Average Recall (0.95) is higher than the Precision, meaning the model is excellent at capturing the correct commands when they occur, even if it occasionally has slight hesitations between similar-sounding labels.

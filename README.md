# Driver's Companion: Drowsiness Detection and Music Recommendation System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains the code and dataset used in the research paper "Driver's Companion: Drowsiness Detection and Emotion Based Music Recommendation System" published in the 2022 International Conference on Computing, Communication, and Intelligent Systems (ICCCIS).

## Introduction

Driver drowsiness is a major cause of road accidents, especially during long journeys. To address this issue, we have developed an intelligent system that can detect driver drowsiness in real-time and recommend suitable music to help the driver stay alert and focused. The proposed system uses computer vision and machine learning techniques to detect the driver's drowsiness level and recommend music based on the driver's emotions.It employs the Eye Aspect Ratio (EAR) for drowsiness detection and a CNN trained on the FER2013 dataset for emotion analysis.

## Methodology

<p align="center">
  <img src="https://user-images.githubusercontent.com/79741191/236828641-b431b81c-06c3-4cec-992a-3e242e30e282.png">
</p>
The methodology combines two components:

1. **Drowsiness Detection**: 
Utilizes dlib's facial landmarks for real-time eye tracking, employing EAR to detect drowsiness.

2. **Emotion-Based Music Recommendation**:
Uses a CNN model for emotion recognition, recommending music based on the Russell’s Circumplex Model of emotions.


## Requirements

The following software is required to run the code in this repository:

- Python 3.6+
- OpenCV 4.0+
- TensorFlow 2.0+
- Keras 2.0+
- NumPy

## Dataset

The FER2013 dataset was used for training the emotion recognition model.

## Usage

1. **Setup and Install Dependencies**: 
   Clone the repository and install the required libraries.
   ```
   git clone https://github.com/shreel143/Drowsiness-Detection-System.git
   cd Drowsiness-Detection-System
   pip install opencv-python tensorflow keras numpy
   ```

2. **Run the Application**: 
   Start the drowsiness detection and emotion-based music recommendation system by running the `drowsiness_detection.py` script.
   ```
   python drowsiness_detection.py
   ```

   Ensure your webcam is properly set up and the environment is well-lit for accurate detection.

The system will start detecting the drowsiness of the driver in real-time using the camera and recommend music based on the driver's emotions.

## Results

The proposed drowsiness detection and music recommendation system achieved an accuracy of 95% on the training data and on test data, accuracy was 83% when trained for 30 epochs. The results and analysis can be found in the research paper.

## Resources

For more in-depth information about the 'Driver's Companion: Drowsiness Detection and Music Recommendation System', please refer to the following resources:

- [Research Paper](https://ieeexplore.ieee.org/document/10037226): Detailed documentation of our research, methodologies, and findings.
- [Project Presentation](https://docs.google.com/presentation/d/1TLywEBDhEMEjHs0gyVwQvLz0fg3Ko8Hu/edit?usp=sharing&ouid=100124470341926430772&rtpof=true&sd=true): A comprehensive presentation that outlines the project's goals, design, and outcomes.

These resources provide a deeper understanding of the project and its impact on driving safety.

## Citation

If you use this code or dataset in your research, please cite the following paper:

```
@inproceedings{pant2022driver,
  title={Driver's Companion: Drowsiness Detection and Emotion Based Music Recommendation System},
  author={Pant, Mridu and Trivedi, Shreel and Aggarwal, Samiksha and Rani, Ritu and Dev, Amita and Bansal, Poonam},
  booktitle={2022 International Conference on Computing, Communication, and Intelligent Systems (ICCCIS)},
  year={2022},
  pages={1-6},
  doi={10.1109/ICCCIS56430.2022.10037226}
}
```

## License

This code is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or comments about this project, do feel free to reach out via [email](mailto:shreektrivedi2020@gmail.com) or connect on [LinkedIn](https://www.linkedin.com/in/shreel-trivedi/).
I am always open to discussing this project, potential collaborations, or any other inquiries you might have.


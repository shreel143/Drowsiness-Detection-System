# Driver's Companion: Drowsiness Detection and Music Recommendation System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains the code and dataset used in the research paper "Driver's Companion: Drowsiness Detection and Emotion Based Music Recommendation System" published in the 2022 International Conference on Computing, Communication, and Intelligent Systems (ICCCIS).

## Introduction

Driver drowsiness is a major cause of road accidents, especially during long journeys. To address this issue, we have developed an intelligent system that can detect driver drowsiness in real-time and recommend suitable music to help the driver stay alert and focused. The proposed system uses computer vision and machine learning techniques to detect the driver's drowsiness level and recommend music based on the driver's emotions.

## Methodology

<p align="center">
  <img src="https://user-images.githubusercontent.com/79741191/236828641-b431b81c-06c3-4cec-992a-3e242e30e282.png">
</p>

## Requirements

The following software is required to run the code in this repository:

- Python 3.6+
- OpenCV 4.0+
- TensorFlow 2.0+
- Keras 2.0+
- NumPy

## Dataset

The dataset used in this research is a combination of publicly available datasets, which can be found in the `dataset` folder.

## Usage

To run the drowsiness detection and music recommendation system, run the `main.py` file using the following command:

```
python main.py
```

The system will start detecting the drowsiness of the driver in real-time using the camera and recommend music based on the driver's emotions.

## Results

The proposed drowsiness detection and music recommendation system achieved an accuracy of 95% on the training data and on test data, accuracy was 83% when trained for 30 epochs.The results and analysis can be found in the research paper.

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

If you have any questions or comments about this project, please feel free to contact the authors. Contact information can be found in the research paper or the GitHub repository.

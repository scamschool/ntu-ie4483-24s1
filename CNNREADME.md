# Cat-Dog Classification Using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) for classifying images of cats and dogs. The model is built and trained using TensorFlow, with the goal of classifying images into two categories: "cat" and "dog".

## Setup

### 1. Create and Activate Conda Environment
To create a new Conda environment with Python 3.8, run the following commands:

```bash
conda create --name catdogenv python=3.8
conda activate catdogenv
```

### 2. Install Required Dependencies
Once the environment is activated, install the necessary Python dependencies using the following commands:
```bash
conda install tensorflow matplotlib pillow numpy pandas scikit-learn
conda install -c conda-forge openpyxl
```

### 3. Clone the Repository
Run the following command to clone this repository to your local workspace:
```bash
git clone https://github.com/scamschool/ntu-ie4483-24s1.git
```

### 4. Organize the Dataset
Make sure the dataset is having the expected structure of the train.py script. The data structure is as shown below:

├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   ├── validation/
│   │   ├── cats/
│   │   └── dogs/

Place the training images of cats in the train/cats/ and validation images of cats in the validation/cats/ directories.
Place the training images of dogs in the train/dogs/ and validation images of dogs in the validation/dogs/ directories.

### 5. Run the Training Script
```bash
python train.py
```

### 6. Run the Testing Script to generate the classification results on the testing dataset
```bash
python test.py
```

An excel file of the classification result will be saved.

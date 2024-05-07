# Young20Percent
 ### Bone Fracture Detection Model

This repository contains the code and resources for a Convolutional Neural Network (CNN) model that can detect and classify bone fractures in X-ray images.

#### File Purpose

- **main.py**: The main entry point of the application. This file defines the CNN architecture, compiles the model, and handles the training and evaluation process.
- **utils.py**: Utility functions for data preprocessing, such as loading and augmenting the image data.
- **data.csv**: The CSV file containing the file paths and class labels for the FracAtlas dataset.
- **README.md**: This file, which provides instructions on how to set up and run the project.

#### Software and Package Installation

1. Install Python (version 3.8 or higher) from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Create a new virtual environment and activate it:
   ```
   python -m venv env
   source env/bin/activate
   ```
3. Install the required packages using the following command:
   ```
   pip install -r requirements.txt
   ```
   The `requirements.txt` file lists all the necessary dependencies for the project, including TensorFlow, Keras, NumPy, and Pandas.

   Link to the FRACATLAS Database: https://figshare.com/articles/dataset/The_dataset/22363012

#### Running the Project

1. Navigate to the project directory.
2. Ensure that your virtual environment is active.
3. Execute the following command to run the project:
   ```
   python main.py
   ```
4. The script will:
   - Load the FracAtlas dataset
   - Preprocess and augment the data
   - Define the CNN architecture
   - Train the model on the training data
   - Evaluate the model's performance on the test data
5. The training and evaluation results will be printed to the console.
(if needed, a full writeup of this project with detailed explanations of the code can be found here: https://docs.google.com/document/d/1NmbWFa3uZH_Kc14jHk9GhjVft4bJDzv_LCiPEw2lMu8/edit?usp=sharing

Here is an image of the first part of the code executing: <img width="700" alt="Screenshot 2024-05-03 at 11 01 41 AM" src="https://github.com/Fyoung24/Young20Percent/assets/95723225/c465a2e7-5f4e-4b52-b2ef-ed3a630d260d">

#### Remaining Bugs and Unimplemented Features

The current implementation of the bone fracture detection model has the following limitations:


1. **Limited Dataset**: The FracAtlas dataset, while comprehensive, may not encompass the full diversity of bone fracture cases encountered in real-world scenarios. Expanding the dataset with additional X-ray images from various sources could improve the model's generalization capabilities.

2. **Does not Work...yet**:I am getting an inexplicable path find error. I do not know why. The first part of the code executes well but it can't find my csv files and I am unsure why. I will continue working on this until I figure it out.

3. **Deployment to Clinical Setting**: To be truly useful in a clinical setting, the model would need to be integrated into a user-friendly interface, with features like real-time prediction, report generation, and seamless integration with existing hospital systems. Developing such a deployment pipeline was beyond the scope of this project.

Future collaborators could address these limitations by:

- Expanding the dataset with more diverse X-ray images
- Implementing explainable AI techniques to provide insights into the model's decision-making
- Developing a clinical deployment pipeline for the bone fracture detection model

By addressing these remaining challenges, the model's accuracy, interpretability, and real-world applicability can be further enhanced, making it a more valuable tool for healthcare professionals in the diagnosis and management of bone fractures.







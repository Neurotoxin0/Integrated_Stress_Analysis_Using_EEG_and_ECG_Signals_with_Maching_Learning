# Integrated Stress Analysis Using EEG and ECG Signals with Machine Learning

## Project Setup and Execution Guide

This README file is designed to guide anyone, including those with no prior knowledge of the project, to successfully set up and run the entire project. Follow the detailed steps and procedures below to ensure a smooth experience.

---

### Hardware and Software Requirements:
- **Hardware:**
  - Memory Requirement: >= 2GB

- **Software:**
  - NodeJS  (version: !!!!!!!!!!!!!!!!!!)
  - Django  (version: !!!!!!!!!!!!!!!!!!)
  - Python  (version: 3.10)

---

### Contributors:

This project was a collaborative effort, and each team member played a crucial role:

- Yang Xu:
  - Machine Learning Algorithm
  - !!!!!!!!!!!

- Xilai Wang:
  - User Friendly Interface
  - !!!!!!!!!!!


---

## Project Overview

This project aims to analyze stress levels by integrating Electroencephalogram (EEG) and Electrocardiogram (ECG) signals using machine learning algorithms. It provides tools for the automated processing and classification of physiological data to identify stress markers. The system includes a user-friendly interface for data input and visualization of analysis results, facilitating research and educational purposes.

---


### Instructions for Setup and Execution:

#### User Friendly Interface

*Further information on the user interface will be provided.*

---

#### Machine Learning Algorithm

- To install required packages, execute `pip install -r requirements.txt`.
- The configuration settings necessary for the application are specified in `backend/config.py`.
- Path configurations have been designed to automatically adjust, eliminating the need for manual changes under standard usage scenarios.
- The model training process is initiated by running `backend/main.py`. This step is crucial for the subsequent analysis.
- The models that have been trained and their corresponding output plots are stored in `backend/Assets/models` and `backend/Assets/plots`, respectively.
- The `run_model.py` script contains a `run` function that processes inputs combining EEG and ECG data. Detailed information on the expected format of the input data can be found by running `test_data_generator.py`.

---

By following these instructions, anyone should be able to successfully set up and run the project. If you encounter any issues or have questions, feel free to contact [provide contact information].

Thank you for using our project!
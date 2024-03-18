# Integrated Stress Analysis Using EEG and ECG Signals with Machine Learning

## Project Setup and Execution Guide

This README file is designed to guide anyone, including those with no prior knowledge of the project, to successfully set up and run the entire project. Follow the detailed steps and procedures below to ensure a smooth experience.

---

### Hardware and Software Requirements:
- **Hardware:**
  - Memory Requirement: >= 2GB

- **Software:**
  - Operation System: Windows 11
  - IDE: Strongly Recommend Visual Studio Code, which is used to develop this project.
  
  ---
  
  **For Front-End Angular Application**
  - (You do not need to install all the packages one by one. You just need to install NodeJS, change to the directory of `/frontend/MindHeartSync` and run `npm install`)
  - NodeJS: v18.19.0
  - Angular: v17.1.3
  - Angular Material: v17.2.1
  - ng-zorro-antd: v17.2.0
  - swiper: v11.0.7
  - Typescript: v5.3.3 (usually automatically installed with Angular)
 
  ---
  
  **For Back-End Django Server:**
  - Django: v5.0.3
  - Python: v3.11.5
 
  --- 
  
  **For Machine Learning Models:**
  - (To install required packages, execute `pip install -r MLalgorithm/requirements.txt`)
  - matplotlib: v3.8.2
  - numpy: 1.22.0
  - pandas: 1.5.3
  - scikit_learn: 1.4.0
  - scikit_optimize: 0.9.0
  - seaborn: 0.13.2
  - tqdm: 4.66.2

---

### Contributors:

This project was a collaborative effort, and each team member played a crucial and even role:

- Yang Xu:
  - Machine Learning Algorithms (Data Processing, Training, Validation, Plot, Export and Import Usage, etc.)

- Xilai Wang:
  - User Friendly Application (Angular Front-End UI and Django Back-End Server)

---

## Project Overview

This project outputs a user-friendly, no coding required application which utilizes machine learning technique to conduct non-clinical, real-time stress assessment using both ECG and EEG siganls as input. It automates the interpretation of physiological data, facilitating easier access to mental health care, for research and educational purpose; And this project integrates multiple types of media to deliver social good.

---


### Instructions for Setup and Execution:

#### User Friendly Application

- The install of the Application can be diveded into 2 phases: installing front-end Angular project and installing back-end Django project.
----1. Front-end Angular project:----
- Install Nodejs v18.19.0
- After successful installation of Nodejs v18.19.0, open PowerShell (you can search for 'powershell' in the bottom of windows 11 screen) as administrator.
- In the Powershell, run this command: npm install -g @angular/cli@17.1.3
- Run Visual Studio Code as administrator (if you do not have, install one)
- File -> Open Folder -> Select the directory of this project (....../Integrated_Stress_Analysis_Using_EEG_and_ECG_Signals_with_Maching_Learning)
- Terminal -> New Terminal. Let's call this terminal terminal 1.
- In terminal 1, run this command: cd frontend/MindHeartSync (going to the root directory of the front-end Angular project)
- In terminal 1, run this command: npm install (automatically installing all packages that this project needs.)
- In terminal 1, run this command: ng serve (run the local server of Angular project)
- Open a browser (recommend Firefox), go to the address shown in terminal 1 (usually http://localhost:4200/ , may have change due to your device's setting)

----2.Back-end Django project:----
- Install Python v3.11.5 in your environment (you can generate a new environment)
- Terminal -> New Terminal. Let's call this terminal terminal 2.
- In terminal 2, run this command: cd backend (going to the root directory of the back-end Django project)
- In terminal 2, run this command: pip install Django==5.0.3
- If the project does not work, in terminal 2, run this command: pip install -r backendrequirements.txt
- In terminal 2, run this command:  python manage.py runserver 8000 (run the local server of Django project)

----If nothing goes wrong, then you can experience the project.----
----Package version mismatch may cause problem using the application.----
----Have you any problem, feel free to contact me in xwang736@uottawa.ca----
----Some customized browser settings and plugins in security aspect might cause failure running this application.----

#### Machine Learning Algorithm

- To install required packages, execute `pip install -r MLalgorithm/requirements.txt`.
- The configuration settings necessary for the application are specified in `MLalgorithm/config.py`.
- Path configurations have been designed to automatically adjust, eliminating the need for manual changes under standard usage scenarios.
- The model training process is initiated by running `MLalgorithm/main.py`. This step is crucial for the subsequent analysis as it generates trained model files for subsequent usage.
- The models that have been trained and their corresponding performance plots will be stored in `MLalgorithm/Assets/models` and `MLalgorithm/Assets/plots`, respectively.
- The `run_model.py` script loads trained standardizing scalar, encoder, PCA model, requested machine learning models respectively, along with processing inputs that combine EEG and ECG data, to predict current estimated stress level.
- Detailed information about the expected format of the input data can be printed by running `test_data_generator.py`.

---

By following these instructions, anyone should be able to successfully set up and run the project. If you encounter any issues or have questions, feel free to contact `[xwang736@uottawa.ca](mailto:xwang736@uottawa.ca)` and `[yxu319@uottawa.ca](mailto:yxu319@uottawa.ca)`

Thank you for using our project!

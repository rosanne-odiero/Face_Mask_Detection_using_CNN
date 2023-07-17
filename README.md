# Face Mask Detection System using Convolutional Neural Network
 This project aims to build a face mask detection system using a Convolutional Neural Network (CNN). The model will be able to predict whether a person is wearing a mask or not based on input images.
 ### Workflow
1. **Data Collection:** Collect an appropriate dataset consisting of two sets of images: people wearing masks and people not wearing masks. This is a binary classification problem, and for this project, we will be using a dataset available on Kaggle.

2. **Image Processing:** Preprocess the image dataset to prepare it for training the CNN. Perform the following processing steps:

- Resize the images to a specific size.
- Convert the images into numpy arrays for easier manipulation.
- Apply any necessary data augmentation techniques.
- Data Splitting: Split the processed images into training and testing datasets. The training set will be used to train the CNN model, while the testing set will be used to evaluate the model's performance.

3. **Convolutional Neural Network:** Feed the training data into a CNN model. The CNN will learn to differentiate between images of people wearing masks and those not wearing masks. This step involves defining the CNN architecture, including layers, filters, and activation functions.

4. **Model Evaluation:** Evaluate the trained CNN model using appropriate evaluation metrics, such as accuracy, precision, recall, and F1 score. This step helps assess the performance and effectiveness of the model in detecting face masks.

5. **Predictive System:** Build a predictive system that utilizes the trained CNN model to make predictions on new images. Given an input image, the system will determine whether the person in the image is wearing a mask or not.

### Repository Structure
- data/: Contains the dataset used for training and testing the CNN.
- Face_Mask_Detection_using_CNN.ipynb: Jupyter Notebook containing the image processing code,CNN model training code, model evaluation code, .
- train_model.ipynb: Jupyter Notebook containing the CNN model training code, demonstrating how to use the trained model for image prediction.
- README.md: Documentation file providing an overview of the project and its workflow.
### Dependencies
Python 3.10
TensorFlow 
NumPy 
OpenCV 
Jupyter Notebook 

### Usage
1. Clone the repository:

`git clone https://github.com/rosanne-odiero/Face_Mask_Detection_using_CNN.git`
2. Install the required dependencies:

`pip install tensorflow numpy opencv-python jupyter`
3. Follow the step-by-step instructions provided in the Jupyter Notebooks to perform image processing, train the CNN model, evaluate its performance, and make predictions on new images.

Credits
[Kaggle Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset?resource=download): Provided is the source and relevant details of the dataset used for this project.








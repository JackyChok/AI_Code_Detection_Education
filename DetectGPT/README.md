# DetectGPT AIGC Detector

The DetectGPT AIGC Detector is a tool designed to detect human-written code in text and merge the results into a CSV file based on code variants. This README provides detailed information on how to set up and use the DetectGPT AIGC Detector.

## Prerequisites

Before you begin, ensure you have access to Google Colaboratory and make sure the runtime type is a GPU or TPU so that CUDA is available for use. The following files should also be ready in your project directory.

- `DetectGPT.ipynb`: The main Jupyter Notebook file for running the DetectGPT AIGC Detector.
- `DetectGPT_postprocessing.ipynb`: The Jupyter Notebook file containing postprocessing methods applied to DetectGPT AIGC Detector outputs.
- `DetectGPT.zip`: DetectGPT model zipped files

In addition to these files, make sure you've updated the paths to the variant folder and AIGC Detector folder for accurate detection.

## Usage

1. **Open the DetectGPT.ipynb file in a Google Colab workspace**
    
    Click on the File tab and click on "Upload Notebook". Drag the DetectGPT.ipynb file to the popup card to upload and open the notebook.

2. **Upload the DetectGPT.zip file to your Google Colab workspace**

    On the left of your screen, open the Files section by clicking the folder icon. Drag the DetectGPT.zip file to the section and wait for the upload to finish.  

3. **Run each line in DetectGPT.ipynb**

    Execute the notebook's cells from top to bottom. Note that you will need to allow Google Colab to mount your Google Drive to store your outputs when the drive.mount('/content/drive') line is executed. 
    
    Be sure to update the path to the variant folder and ensure that the input variant CSV file is located in the correct pathway within your project directory.

4. **Run DetectGPT_postprocessing.ipynb with DetectGPT Output**

    Execute the notebook's cells from top to bottom. This notebook should be executed on your local machine instead of Google Colab due to different file access methods.
    
    Be sure to update the path to the DetectGPT output folder and results folder.


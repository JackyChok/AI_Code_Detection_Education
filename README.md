# Assessing AI Detectors in Identifying AI-Generated Code: Implications for Education

# Abstract
This abstract introduces the artifact associated with the ICSE '24 paper titled "Assessing AI Detectors in Identifying AI-Generated Code: Implications for Education." The study delves into an empirical examination of the LLM's attempts to circumvent detection by AIGC Detectors. The methodology entails code generation in response to targeted queries using diverse variants. Our primary goal is to attain the Available and Reusable badges. The abstract further offers comprehensive technical details about each artifact component and elucidates its utility for prospective research endeavors.

# Research Question
The following two research questions were constructed to guide the study.
- RQ1: *How accurate are existing AIGC Detectors at detecting AI-generated code?*
- RQ2: *What are the limitations of existing AIGC Detectors when it comes to detecting AI-generated code?*

# Replication of the empirical study
We have provided the 13 variants of AI-Generated Content (AIGC) in data folder. Each variant will be using the same set of human-written code, hence, you will only have to execute the code detection for human-written code once.

README file are provided for each AIGC Detector, also known as Code Detection Model (CDM), to provide guidance on how to setup the AIGC Detector. Then, you can execute the code detection file for each AIGC Detector and get the results for the corresponding variant.

After retrieving all the results from the AIGC Detector, execute the code in _metrics_performance.ipynb to get the performance of the AIGC Detector. You are recommended to combine all results csv file into one file, with one header and 13 variant of detector classification results.

The prompt details of each variant can be found [here](https://figshare.com/articles/dataset/Variant_Description/24265018). For variant 8, 9, 10, we have developed python code to replace the AIGC.

# Folder structure
The project directory should contain the following structure:
- `GPTZero` - Folder that contains all files required to setup and run GPTZero
- `DectectGPT` - Folder that contains all files required to setup and run DectectGPT
- `GLTR` - Folder that contains all files required to setup and run GLTR
- `Sapling` - Folder that contains all files required to setup and run Sapling
- `Gpt2outputdetector` - Folder that contains all files required to setup and run GPT-2 Output Detector
- `VariantData` - Folder that contains all variant data with the AIGC, there should be 13 variant file 
- `_metrics_performance.ipynb` - Python jupyter notebook that is used to calculate the performance of the AIGC Detector
- `.gitattributes` - For upload large files like ".pt"
- `STATUS.md` - For mentions the badges we are applying
- `LICNSE` - For describing the distribution rights

# Comprehensive Results Overview: AIGC Detector Variants
The summarized results for each variant of AIGC Detector can be found in the respective results folders within each AIGC Detector directory.

# Coding Guide for RQ1
Below show the detailed instructions of each AIGC Detector on accessing the results of each variant. Please direct to each AIGc Detector project directory first then only follows the steps on each AIGC Detectors below. After obtaining the results, follow the example provided in `_metrics_performance.ipynb` to calculate the metrics performance of the AIGC Detector.

## 1. DetectGPT AIGC Detector

The DetectGPT AIGC Detector is a tool designed to detect human-written code in text and merge the results into a CSV file based on code variants. Below provides detailed information on how to set up and use the DetectGPT AIGC Detector.

### Prerequisites

Before you begin, ensure you have access to Google Colaboratory and make sure the runtime type is a GPU or TPU so that CUDA is available for use. The following files should also be ready in the DetectGPT project directory.

- `DetectGPT.ipynb`: The main Jupyter Notebook file for running the DetectGPT AIGC Detector.
- `DetectGPT_postprocessing.ipynb`: The Jupyter Notebook file containing postprocessing methods applied to DetectGPT AIGC Detector outputs.
- `DetectGPT.zip`: DetectGPT model zipped files

In addition to these files, make sure you've updated the paths to the variant folder and AIGC Detector folder for accurate detection.

### Usage

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

## 2. GLTR AIGC Detector

The GLTR AIGC Detector is a tool designed to detect human-written code in text and merge the results into a CSV file based on code variants. Below provides detailed information on how to set up and use the GLTR AIGC Detector.

### Prerequisites

Before you begin, ensure you have the following files and dependencies ready in the GLTR project directory:

- `cdm_automation.ipynb`: The main Jupyter Notebook file for running the GLTR AIGC Detector.
- `cdm.py`: Configuration file for the GLTR AIGC Detector.
- `en-gpt2-gltr.pkl`: GLTR model file.
- `en-gpt2-ppl.pkl`: PPLM model file.
- `english.pickle`: Pickled language model file.

In addition to these files, make sure you've updated the paths to the variant folder and AIGC Detector folder for accurate detection.

### Usage

1. **Run the Jupyter Notebook:**

   Open `cdm_automation.ipynb` in Jupyter Notebook and execute the cells. This notebook applies the GLTR AIGC Detector and generates a CSV file containing the detected human-written and AI-generated code results. Be sure to update the path to the variant folder and ensure that the input variant CSV file is located in the correct pathway within your project directory.

Please make sure to update the relevant paths and file locations as needed.

## 3. GPT-2 Output Detector
Below provides guidance on how to setup and execute code detection for this AIGC Detector.

### Prerequisites
Before you begin, ensure you have the following files and dependencies ready in your project directory:

- `cdm_automation.ipynb`: The main Jupyter notebook file for running the GPT-2 output detector
- `gpt2_output_detector.py`: Configuration file for the GPT-2 output detector
- `Setup`: Folder containing all required files to setup the GPT-2 output detector

In addition to these files, make sure you've updated the paths to the variant folder and AIGC Detector folder.
### Usage
#### Setup
The in-depth details on how to setup GPT-2 Output Detector can be found on Setup folder.

#### Code Dectection
1. **Run the Jupyter Notebook:**

    Open `cdm_automation.ipynb` in Jupyter Notebook and execute the cells. This notebook applies the GPT-2 output detector and generates a CSV file containing the detected human-written and AI-generated code results. Be sure to update the path to the variant folder and ensure that the input variant CSV file is located in the correct pathway within your project directory.

Please make sure to update the relevant paths and file locations as needed.


## 4. GPTzero AIGC Detector
GPTZero is an AI-generated content detection system developed to identify and analyze artificially generated text, specifically focusing on text produced by AI language models such as GPT-3. Below provides detailed information on how to set up and use the GPTzero AIGC Detector.

### Prerequisites
Before you begin, ensure you acquired the API key from [GPTZero](https://gptzero.me/):

- `GPTzero_cdm_detect.ipynb`: The main Jupyter Notebook file for running the GLTR AIGC Detector.

In addition to these files, make sure you've updated the paths to the variant folder and AIGC Detector folder for accurate detection.

### Usage

1. **Run the Jupyter Notebook:**

   Open the `GPTzero_cdm_detect.ipynb` in Jupyter Notebook and execute the cells as per the provided documentation in the file. After running the file, it will call the GPTzero API and generate a CSV file with the detected human-written and AI-generated code results. Ensure that you update the path to the variant folder and confirm that the input variant CSV file is in the correct location within your project directory.

Please make sure to update the relevant paths and file locations as needed.

## 5. Sapling Detector

### Getting Started

This project uses the Sapling API to evaluate code. Follow the steps below to get started:

#### Prerequisites

Before using this project, you'll need:

- Python (version 3.9)
- Pip (for installing Python packages)
- A Sapling API key (obtain from [Sapling API Website](https://sapling.ai))


# Coding Guide for RQ2
Follow the same procedure as outlined in the RQ1 Coding Guide, but additionally, execute the Statistical Test demonstrated in the  `_metrics_performance.ipynb` for a comprehensive analysis.

# Authors
- Wei Hung Pan
- Ming Jie Chok
- Jonathan Wong Leong Shan
- Yung Xin Shin
- Yeong Shian Poon
- Zhou Yang
- Chun Yong Chong
- David Lo
- Mei Kuan Lim
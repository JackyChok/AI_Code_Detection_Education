# GLTR AIGC Detector

The GLTR AIGC Detector is a tool designed to detect human-written code in text and merge the results into a CSV file based on code variants. This README provides detailed information on how to set up and use the GLTR AIGC Detector.

## Prerequisites

Before you begin, ensure you have the following files and dependencies ready in your project directory:

- `cdm_automation.ipynb`: The main Jupyter Notebook file for running the GLTR AIGC Detector.
- `cdm.py`: Configuration file for the GLTR AIGC Detector.
- `en-gpt2-gltr.pkl`: GLTR model file.
- `en-gpt2-ppl.pkl`: PPLM model file.
- `english.pickle`: Pickled language model file.

In addition to these files, make sure you've updated the paths to the variant folder and AIGC Detector folder for accurate detection.

## Usage

1. **Run the Jupyter Notebook:**

   Open `cdm_automation.ipynb` in Jupyter Notebook and execute the cells. This notebook applies the GLTR AIGC Detector and generates a CSV file containing the detected human-written and AI-generated code results. Be sure to update the path to the variant folder and ensure that the input variant CSV file is located in the correct pathway within your project directory.

Please make sure to update the relevant paths and file locations as needed.

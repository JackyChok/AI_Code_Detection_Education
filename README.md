# Assessing AI Detectors in Identifying AI-Generated Code: Implications for Education

This is a research artifact for the ICSE 2024 paper. The following two research questions were constructed to guide the study.
- RQ1: *How accurate are existing AIGC Detectors at detecting AI-generated code?*
- RQ2: *What are the limitations of existing AIGC Detectors when it comes to detecting AI-generated code?*

# Replication of the empirical study
We have provided the 13 variants of AI-Generated Content (AIGC) in data folder. Each variant will be using the same set of human-written code, hence, you will only have to execute the code detection for human-written code once.

README file are provided for each AIGC Detector, also known as Code Detection Model (CDM), to provide guidance on how to setup the AIGC Detector. Then, you can execute the code detection file for each AIGC Detector and get the results for the corresponding variant.

After retrieving all the results from the AIGC Detector, execute the code in _metrics_performance.ipynb to get the performance of the AIGC Detector. You are recommended to combine all results csv file into one file, with one header and 13 variant of detector classification results.

The prompt details of each variant can be found [here](https://figshare.com/articles/dataset/Variant_Description/24265018). For variant 8, 9, 10, we have developed python code to replace the AIGC.

# Comprehensive Results Overview: AIGC Detector Variants
The summarized results for each variant of AIGC Detector can be found in the respective results folders within each AIGC Detector directory.

# Coding Guide for RQ2
Refer to the README file in each AIGC Detector directory for detailed instructions on accessing the results of each variant. After obtaining the results, follow the example provided in `_metrics_performance.ipynb` to calculate the metrics performance of the AIGC Detector.

# Coding Guide for RQ3
Follow the same procedure as outlined in the RQ2 Coding Guide, but additionally, execute the Statistical Test demonstrated in the  `_metrics_performance.ipynb` for a comprehensive analysis.

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
- `STATUS.md` - For mentions what badges we are applying
- `LICNSE` - For describing the distribution rights

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
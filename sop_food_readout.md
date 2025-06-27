This is an overview of the standard operating procedure (SOP) for generating food readouts from untargeted metabolomics data using GNPS2 and Food-MetaboApp. This SOP is designed to help researchers identify food molecules in their datasets and generate dietary intake readouts.

**To use this app ([Step 3](#step-3-food-readout-generation-with-food-metaboapp))**, you will need to have completed the following steps:

### **Step 1: Traditional Metabolomics Feature Extraction :** 
Perform feature extraction using your preferred software (e.g., MZmine) and generate the required output files:

* Spectral information file: `.mgf` format (e.g., `filename_iimn_gnps.mgf`)  
* Quantification data file: `.csv` format (e.g., `filename_iimn_gnps_quant.csv`)

These files contain the mass spectrometry data needed for downstream food metabolite identification.

### **Step 2: GNPS Food Library search Workflow :** 
Identify food molecules in your dataset using the GNPS2 library search workflow:

1. Navigate to GNPS2 and select the Library Search Workflow  
2. Configure the following inputs:  
   * Input Data Folder: Upload your `filename_iimn_gnps.mgf` file from Step 1  
   * Input Library Folder: Upload the food dataset MGF file (`500_foods_Spectrum.mgf` from our GitHub repository)  
   * Other parameters: Leave at default values  
   * Analog search: Set to "Yes" if analyzing stool samples (recommended for detecting food molecule analogs)  
3. Click Submit Workflow 

### **Step 3: Food Readout Generation with Food-MetaboApp :** 
Generate dietary intake readouts from your metabolomics data:

1. Input the GNPS library search task ID from Step 2  
2. Select food ontology level for dietary intake analysis (refer to Foodomics paper citation)  
3. Upload quantification data using one of these methods:  
   * Direct upload: Use `filename_iimn_gnps_quant.csv` from Step 1  
   * GNPS2 integration: Provide your feature-based molecular networking job task ID (software will automatically fetch quantification files)  
4. Click Run Analysis to generate dietary readout

### **Step 4: Downstream Data Analysis :** 
Perform statistical analysis and visualization of your food metabolomics results:

* Upload metadata file with shared "filename" column matching your quantification table  
* Perform dimensionality reduction analysis between groups (PCA) on food readout data, Conduct univariate analysis on dietary intake using metadata variables of interest

### **External Analysis Options**

* Download food readout files (CSV/TSV format)  
* Perform custom data analysis using R or Python, Generate publication-ready visualizations and statistical summaries


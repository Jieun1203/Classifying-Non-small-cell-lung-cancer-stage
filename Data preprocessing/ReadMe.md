#README

Codes for CT data preprocessing

For fair comparison, isotropic resamplig is needed when CT images were obtained from various CT modalities.

CT images were .gz file format.
CT images were cropped into 128x128x3 pathes centering around the most biggest ROI slice.
Firstly, resampled CT images and ROI images were saved in the folder 'resampled' in data_root folder.
Second, CT images were cropped into 128x128x3 pathes centering around the most biggest ROI slice and saved in 'numpy' folder.
Lastly, numpy files are normalized and saved in 'numpy' folder.


<pre>
<code>
#how to use
python3 CT_normalization.py --data_root /DATA/data_cancer/NSCLC_Radiogenomics
</code>
</pre>
#please change the data_root according to datasets

Since clinical information of the datasets are given in different setting, stages should be extracted in handed-manner.

In the study, according to patients number, resampled and normalized ct images are combined into one numpy file.

Also in same order, stage information was made in a single numpy file.



In the study, 6 datasets were used.
1. NSCLC-Radiogenomics
2. NSCLC-Radiomic-Genomics
3. CPTAC-LUAD
4. CPTAC-LSCC
5. TCGA-LUAD
6. TCGA-LUSC

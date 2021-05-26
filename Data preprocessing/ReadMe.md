#README

Codes for CT data preprocessing
For fair comparison, isotropic resamplig is needed when CT images were obtained from various CT modalities.

CT images were .gz file format.

<pre>
<code>
#how to use
python3 CT_normalization.py --data_root /DATA/data_cancer/NSCLC_Radiogenomics
</code>
</pre>



In the study, 6 datasets were used.
1. NSCLC-Radiogenomics
2. NSCLC-Radiomic-Genomics
3. CPTAC-LUAD
4. CPTAC-LSCC
5. TCGA-LUAD
6. TCGA-LUSC

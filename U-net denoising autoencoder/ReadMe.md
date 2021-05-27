#ReadME

Codes for Training denoising U-net shape autoencoder

Normalized CT image and stage files are should be in numpy file format in each datasets.  
Random Gaussian noise will be added to original input image and trained to reconstruct the original image without noise.  
Latent variable is extracted from the bottel neck layer of autoencoder.  

After the training, 
1. original CT images
2. noise added CT images
3. reconstructed CT images
4. stages
5. latent variables are saved.
6. weights of autoencoder
will be saved at save_root folder

<pre>
<code>
#how to use
python Encoder_training.py --train_image_root /nsclc/nsclc_radiogenomics/ct_total.npy --train_stage_root /nsclc/nsclc_radiogenomics/stage_total.npy --val_image_root /nsclc/nsclc_ragiomics_genomics/ct_total.npy --val_stage_root /nsclc/nsclc_ragiomics_genomics/stage_total.npy --test_image_root1 /nsclc/CPTAC_LUAD/ct_total.npy --test_stage_root1 /nsclc/CPTAC_LUAD/stage_total.npy --test_image_root2 /nsclc/CPTAC_LUSC/ct_total.npy --test_stage_root2 /nsclc/CPTAC_LUSC/stage_total.npy --independent_val No --save_root /nsclc/save_root
</code>
</pre>


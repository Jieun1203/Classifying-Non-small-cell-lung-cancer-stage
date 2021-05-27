#READ ME

Codes for NSCLC overall pathological stage classification  

Latent variable extracted from autoencoder is utilized as input in this classification network
Models will be training during the number of epoch as set by user.

*caution: the data_root option in here should be same as the save_root which were setted when training autoencoder.

<pre>
<code>
#how to use
python Encoder_training.py --data_root /nsclc/save_root --save_root /nsclc/final_save_root
</code>
</pre>

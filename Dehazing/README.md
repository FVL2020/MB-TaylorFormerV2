## Prepare dataset for training and evaluation
The `data` directory structure will be arranged as: 
```
data
   |- ITS
      |- Train
      	|- Haze
        	|- 1_1_0.90179.png 
        	|- 2_1_0.99082.png
        |- GT
        	|- 1.png 
        	|- 2.png
      |- Test
      	|- Haze
        	|- 00001.png
        	|- 00002.png
        |- GT
        	|- 00001.png
        	|- 00002.png
   |- OTS
      |- Train
      	|- Haze
        	|- 0001_0.85_0.04.jpg
        	|- 0002_0.85_0.04.jpg
        |- GT
        	|- 0001.jpg
        	|- 0002.jpg
      |- Test
      	|- Haze
        	|- 00501.png
        	|- 00502.png
        |- GT
        	|- 00501.png
        	|- 00502.png
   |- Dense-Haze
      |- Train
         |- Haze
            |- 01_hazy.png 
            |- 02_hazy.png
         |- GT
            |- 01_GT.png 
            |- 02_GT.png
      |- Test
         |- Haze
            |- 51_hazy.png 
            |- 52_hazy.png
         |- GT
            |- 51_GT.png 
            |- 52_GT.png
   |- NH-Haze
      |- Train
         |- Haze
            |- 01_hazy.png 
            |- 02_hazy.png
         |- GT
            |- 01_GT.png 
            |- 02_GT.png
      |- Test
         |- Haze
            |- 51_hazy.png 
            |- 52_hazy.png
         |- GT
            |- 51_GT.png 
            |- 52_GT.png
   |- O-HAZE
      |- Train
         |- Haze
            |- 01_outdoor_hazy.jpg
            |- 02_outdoor_hazy.jpg
         |- GT
            |- 01_outdoor_GT.jpg 
            |- 02_outdoor_GT.jpg
      |- Test
         |- Haze
            |- 41_outdoor_haze.jpg
            |- 42_outdoor_haze.jpg
         |- GT
            |- 41_outdoor_GT.jpg
            |- 42_outdoor_GT.jpg
   |- Haze4K
      |- Train
         |- Haze
            |- 1000_0.74_1.6.png
            |- 1001_0.89_1.66.png
         |- GT
            |- 1000.png
            |- 1001.png
      |- Test
         |- Haze
            |- 1000_0.73_1.8.png
            |- 100_0.5_1.76.png
         |- GT
            |- 1000.png
            |- 100.png


```

## Training
Please modify MB-TaylorFormer-B.yml or MB-TaylorFormer-L.yml first.

To train MB-TaylorFormerV2, run

```
sh /train.sh Dehazing/Options/MB-TaylorFormer-B.yml 
```

or

```
sh /train.sh Dehazing/Options/MB-TaylorFormer-L.yml
```

## Testing

Run the following command to quick test:

```shell
CUDA_VISIBLE_DEVICES=X python Dehazing/test.py --size ['B' or 'L'] --input_dir [Input path] --target_dir [GT path] --result_dir [Result path] --weights [Model weighting path]
```

For example:

```
CUDA_VISIBLE_DEVICES=0 python Dehazing/test.py --size 'B'  --input_dir './Dehazing/datasets/ITS/test/Haze/' --target_dir './Dehazing/datasets/ITS/test/GT/' --result_dir './ITS_result' --weights './Dehazing/pretrained_models/ITS-B.pth'
```


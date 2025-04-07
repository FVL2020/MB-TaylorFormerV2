## Prepare dataset for training and evaluation
The `data` directory structure will be arranged as: 
```
data
   |- SRRS
      |- train2500
        |- Snow
        	|- 0.tif
        	|- 10006.tif
        |- Gt
        	|- 0.jpg
        	|- 10006.jpg
      |- test2000
        |- Snow
        	|- 1.tif
        	|- 101.tif
        |- Gt
        	|- 1.jpg
        	|- 101.jpg
   |- Snow100K
      |- train2500
        |- Snow
        	|- beautiful_smile_00011.jpg
        	|- beautiful_smile_00014.jpg
        |- Gt
        	|- beautiful_smile_00011.jpg
        	|- beautiful_smile_00014.jpg
      |- test2000
        |- Haze
        	|- beautiful_smile_00004.jpg
        	|- beautiful_smile_00052.jpg
        |- GT
        	|- beautiful_smile_00004.jpg
        	|- beautiful_smile_00052.jpg

```

## Training
Please modify MB-TaylorFormer-L.yml first.

To train MB-TaylorFormerV2, run
```
sh /train.sh Dehazing/Options/MB-TaylorFormer-L.yml
```

## Testing

Run the following command to quick test:

```shell
CUDA_VISIBLE_DEVICES=X python Desnowing/test.py --input_dir [Input path] --target_dir [GT path] --result_dir [Result path] --weights [Model weighting path]
```

For example:

```
CUDA_VISIBLE_DEVICES=0 python Desnowing/test.py  --input_dir './Desnowing/datasets/SRRS/test2000/Snow/' --target_dir './Desnowing/datasets/SRRS/test2000/Gt/' --result_dir './SRRS_result' --weights './Desnowing/pretrained_models/SRRS-B.pth'
```


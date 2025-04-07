To download GoPro training and testing data, run
```
python download_data.py --data train-test
```

2. Generate image patches from full-resolution training images of GoPro dataset
```
python generate_patches_gopro.py 
```

## Training
Please modify MB-TaylorFormer-XL.yml first.

To train MB-TaylorFormerV2, run
```
sh /train.sh Dehazing/Options/MB-TaylorFormer-XL.yml
```

## Testing

Download test datasets (GoPro, HIDE), run 
```
python download_data.py --data test --dataset HIDE
```
or

```
python download_data.py --data test --dataset GoPro
```

Run the following command to quick test:

```shell
CUDA_VISIBLE_DEVICES=X python Deblurring/test.py  --input_dir [Input path] --result_dir [Result path] --weights [Model weighting path]
```

For example:

```
CUDA_VISIBLE_DEVICES=0 python Deblurring/test.py  --input_dir './Deblurring/datasets/HIDE/test/blur/' --result_dir './HIDE_result' --weights './Deblurring/pretrained_models/HIDE-XL.pth'
```

#### To reproduce PSNR/SSIM scores of Table 1, run

```
evaluate_gopro_hide.m 
```




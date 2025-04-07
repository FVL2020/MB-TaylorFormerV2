
To download Rain13K training and testing data, run
```
python download_data.py --data train-test
```

## Training
Please modify MB-TaylorFormer-L.yml first.

To train MB-TaylorFormerV2, run
```
sh /train.sh Dehazing/Options/MB-TaylorFormer-L.yml
```

## Testing

Download test datasets (Test100, Rain100H, Rain100L, Test1200, Test2800), run 
```
python download_data.py --data test
```

Run the following command to quick test:

```shell
CUDA_VISIBLE_DEVICES=X python Deraining/test.py  --input_dir [Input path] --result_dir [Result path] --weights [Model weighting path]
```

For example:

```
CUDA_VISIBLE_DEVICES=0 python Deraining/test.py  --input_dir './Deraining/datasets/Rain13K/test/rain/'  --result_dir './Rain13K_result' --weights './Draining/pretrained_models/DerainV2-L.pth'
```

#### To reproduce PSNR/SSIM scores of Table 1, run

```
evaluate_PSNR_SSIM.m 
```




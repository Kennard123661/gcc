To run pretraining for OGO:
```
bash ogo/pretrain.sh 2 --batch-size [batchsize] --num_graphs [number of grapsh per batch, K]
```
Some examples include:
```
bash ogo/pretrain.sh 2 --batch-size 32 --num_graphs 5
```

Then, to run finetuning, we follow the same as the original code:
```
bash ogo/finetune.sh <load_path> <gpu> usa_airport
```

# Odd-Graph-Out Pretraining

This reapdme describes how to run our code:

## Setting up

Create a conda environment and run the following commands.

```
conda create -n gcc python==3.7.6
conda activate gcc
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html  
pip install dgl-cu101==0.4.3.post2  # for data augmentation of the graphs
pip install -r requirements.txt
conda install -c conda-forge rdkit
``` 

If your system does not have space to download the datasets on the `/home` directory and you have hdd, run the following
```
export DGL_DOWNLOAD_DIR=[SAVE_DIR]  # replace SAVE_DIR with your desired directory.
```

Download all pretraining datasets with the following command:
```
python scripts/download.py --url https://drive.google.com/open?id=1JCHm39rf7HAJSp-1755wa32ToHCn2Twz --path data --fname small.bin
```

Download all downstream datasets with the following command:
```
python scripts/download.py --url https://drive.google.com/open?id=12kmPV3XjVufxbIVNx5BQr-CFM9SmaFvM --path data --fname downstream.tar.gz
```


## Pretraining

To run pretraining for OGO:
```
bash ogo/pretrain.sh 2 --batch-size [batchsize] --num_graphs [number of grapsh per batch, K]
```
Some examples include:
```
bash ogo/pretrain.sh 2 --batch-size 32 --num_graphs 5
```
which pretrains a GIN using OGO with batchsize of 32 and the number of graphs K is 5

## Finetuning

Then, to run finetuning on a pretrained OGO network with num_graphs=3 (for num_graphs=5, run `ogo/finetune_5.sh`, run the following code:
```
bash ogo/finetune_3.sh <load_path> <gpu> usa_airport
```
The datasets that we have are `usa_airport`, `h-index`, `imdb-binary`, `imdb-multi`, `collab`, `rdt-b`, `rdt-5k`. 


## Pretraining GCC

To finetune the pretrain network, run the following command:
```
bash scripts/pretrain.sh --batch-size 32
```
which pretrains GIN using the GCC pretraining.
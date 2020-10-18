## Setting up


```
conda create -n gcc-moco python==3.7.6
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl-cu101==0.4.3.post2

pip install -r requirements.txt
conda install -c conda-forge rdkit
conda install -c dglteam dgl-cuda10.1 
```
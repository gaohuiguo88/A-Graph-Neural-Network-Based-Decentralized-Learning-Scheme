1. prepare the environment
open anaconda prompt, type:
conda create --name gnn python=3.9.0
conda activate gnn
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric

2. run generate_dataset.py to generate the dataset

3. run TrainTest_GNNandFIR.py to train GNN and FIR and generate the test data simultaneously

4. After generating the test data, run generate_NodeEdgeRemoval_dataset.py to generate new test data.

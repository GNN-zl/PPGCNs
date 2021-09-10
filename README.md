# PPGCNs
This is the code of paper: Proximity Preserving Graph Convolution Networks.


## Requirements
* Python 3.6.12
* Please install other pakeages by 
``` pip install -r requirements.txt```

## Usage Example
* Running on cora, citeseer, and pubmed:
```sh semi.sh ```


## Results

Our model achieves the following accuracies on Cora, CiteSeer and Pubmed with the public splits:

| Model name   |   Cora    |  CiteSeer |  Pubmed   |
| ------------ | --------- | --------- | --------- |
| SGC-Lpp(k=1) |   84.0%   |    73.4%  |   80.3%   |

## Running Environment 

The experimental results reported in paper are conducted on a single NVIDIA GeForce RTX 2080 Ti with CUDA 10.2, which might be slightly inconsistent with the results induced by other platforms.

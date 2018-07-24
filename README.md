# Adverserial Inductive Deep Walk (AIDW)
AIDW exploits the strengths of generative adversarial networks in capturing latent features, and learns stable and robust graph representations. It consists of two components, i.e., a structure preserving component and an adversarial learning component. The former component aims to capture network structural properties, while the latter contributes to learning robust representations by matching the posterior distribution of the latent representations to given priors.

<img src="/images/biLSTM.png" width="70%">

## Installation
`pip3 install -r requirements.txt`    
Dataset : [citeseer](http://citeseerx.ist.psu.edu/index) Included [here](/)

## Usage
Default paramenters - `python3 main.py`  
man page - `python3 main.py -h`

## Credits
- [Dr. Elena Zheleva](https://www.cs.uic.edu/~elena/)
- Quanyu Dai, Qiang Li, Jian Tang, Dan Wang. 2017. [_Adversarial Network Embedding_](https://arxiv.org/pdf/1711.07838.pdf)


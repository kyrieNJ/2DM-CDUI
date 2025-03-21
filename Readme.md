# 2DM-CDUI
Code for our paper:
Enhancing Depression Detection with Cognitive Distortion and User-Level Information


## Requirements
* Python 3.8
* torch 2.3.0
* SpaCy 3.7.2
* numpy 1.24.3
* argparse 1.4.0
* scikit-learn 1.3.2
* transformers 4.34.1
* zh-core-web-sm 3.7.0
* jieba 0.42.1


## Multi-granularity Cognitive Distortion Learning Method

* Process the data in C2D2 into the data samples needed for contrastive learning, run the code [processdata4cd.py](./mcdlm/processdata4cd.py).
* Train by cognitive distortion type prediction task, run the code [train4cd.py](./mcdlm/train4cd.py).
* Train by scene contrastive pre-training task, run the code [train4CL.py](./mcdlm/train4CL.py).
* Generate cognitive distortion embeddings using the trained model, run the code [infer4cd.py](./mcdlm/infer4cd.py).

## Data pre-processing stage

* Generate user-level information, run the code [user_BehaProf_ana.py](./user_BehaProf_ana.py).

## Train stage
* You can train the model, run the code [train.py](./train.py).
```bash
python ./train.py 
```


## Citation

If our work has been helpful to you, please mark references to our work in your research and thank you for your support.


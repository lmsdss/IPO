# IPO: Interpretable Prompt Optimization for Vision-Language Models(NeurIPS 2024)

[[`arxiv`](https://arxiv.org/abs/2410.15397)] 


## How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to Run
To start training the model, you can run the following script:  
bash scripts/IPO/train.sh

To test the model, you can run the following script:  
bash scripts/IPO/test.sh

To generate image descriptions using MiniCPM-V-2.0, run the following script:  
python generate_descriptions.py


## Acknowledgements
We gratefully acknowledge that the IPO code is based on the excellent repositories [CoOp](https://github.com/KaiyangZhou/CoOp/tree/main) and [OPRO](https://github.com/google-deepmind/opro). Many thanks to their contributors for their inspiring work!

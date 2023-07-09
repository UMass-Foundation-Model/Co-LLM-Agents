# Building Cooperative Embodied Agents Modularly with Large Language Models

This repo contains codes for the following paper:

_Hongxin Zhang*, Weihua Du*, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B. Tenenbaum, Tianmin Shu, Chuang Gan_: Building Cooperative Embodied Agents Modularly with Large Language Models 

Paper: [Arxiv](https://arxiv.org/abs/2307.02485)

Project Website: [Co-LLM-Agents](https://vis-www.cs.umass.edu/Co-LLM-Agents/)

![Pipeline](assets/pipeline.png)

## Installation

For instructions on the installation of the two embodied multi-agent environments `Communicative Watch-And-Help` and `ThreeDWorld Multi-Agent Transport`, please refer to the Setup sections in `envs/cwah/README.md` and `envs/tdw_mat/README.md`, respectively.

## Run Experiments

The main implementation code of our Cooperative LLM Agents is in `envs/*/LLM` and `envs/*/lm_agent.py`. The scripts for running the agents are in `envs/*/scripts` folder, you can try to run them on your own.

When running the scripts, be sure that you are in the parent folder of `scripts` (i.e., `envs/tdw_mat` or `envs/cwah`).

## Citation
If you find our work useful, please consider citing:
```
@article{zhang2023building,
  title={Building Cooperative Embodied Agents Modularly with Large Language Models},
  author={Zhang, Hongxin and Du, Weihua and Shan, Jiaming and Zhou, Qinhong and Du, Yilun and Tenenbaum, Joshua B and Shu, Tianmin and Gan, Chuang},
  journal={arXiv preprint arXiv:2307.02485},
  year={2023}
}
```
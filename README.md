# Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning

This is the implementation for the paper [Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning](https://arxiv.org/pdf/2410.19290).
We propose a fine-tuning strategy called PREREQ-TUNE to reduce LLM hallucinations. PREREQ-TUNE disentangles the learning of skills and knowledge, addressing the knowledge inconsistency between pre-training and fine-tuning. It further leverages fictitious synthetic data to enhance the grounding of LLM outputs to their internal knowledge.


## Quick Links
- [**Dataset on Hugging Face**](https://huggingface.co/datasets/Shiyu-Lab/Prereq_Tune): Link to our synthetic datasets.
- [**Models on Hugging Face**](https://huggingface.co/collections/Shiyu-Lab/prereq-tune-models-677f802408c1365f5557d3b9): Link to our fine-tuned models.


## Installation
To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n prereq_tune python=3.10 && conda activate prereq_tune
```

Next, install PyTorch `v2.4.0`:
```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

You can then install the remaining package dependencies as follows:

```shell
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
rm setup.py
cp ../setup.py ./
python -m pip install .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn==2.6.3 --no-build-isolation
```


## Usage

### Code Structure
```
├── bash_scripts                 <- Bash scripts to run experiments
├── recipes                      <- Recipe configs for all datasets, accelerate configs
├── scripts                     
│   ├── run_cpt.py               <- First step prerequisite learning
│   ├── run_sft.py               <- Second step supervised fine-tuning
│   ├── evaluate_qa.py           <- Evaluate on PopQA and HotpotQA
│   ├── generate_longform.py     <- Generate long-form answers
```

### Run Experiments
To run our experiments, use the corresponding script in `bash_scripts`.

For example, for HotpotQA, you can run:
```shell
bash bash_scripts/hotpotqa.sh 
```
Please remember to replace `SAVE_DIR` with your specific path to save the models.

### Evaluation
The evaluation for PopQA and HotpotQA is already included in their bash scripts, please refer to the scripts for details.

For biography generation and medical QA, we use [FActScore](https://github.com/shmsw25/FActScore) for evaluation. However, we slightly modify the pipeline of FActScore as described in our paper. We will release the modified code soon.


## Citation
If you find the content of this repo useful in your work, please cite it as follows:

```bibtex
@misc{liu2024fictitioussyntheticdataimprove,
      title={Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning}, 
      author={Yujian Liu and Shiyu Chang and Tommi Jaakkola and Yang Zhang},
      year={2024},
      eprint={2410.19290},
      archivePrefix={arXiv},
      primaryClass={cs.CL}, 
}
```

## Acknowledgement
Our implementation is based on the following repos:
* [https://github.com/huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook)

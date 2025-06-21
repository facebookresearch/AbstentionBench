<div align="center">

  # AbstentionBench: A Holistic Benchmark for LLM Abstention


[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.09038)

</div>

<div align="center" style="font-family: Arial, sans-serif;">
  
  <p>
    <a href="#installation" style="text-decoration: none; font-weight: bold;">🔧 Installation</a> •
    <a href="#run-experiments" style="text-decoration: none; font-weight: bold;">✨ Evaluate Abstention</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🖋️ Citation</a>
  </p>
</div>



For reliable LLM deployment, knowing when _not_ to answer is just as important as answering correctly.
Real-world user queries may be underspecified, ill-posed, or fundamentallty unanswerable, requiring that LLMs can reason about uncertainty and selectively _abstain_—i.e., refuse to answer definitively. 

**AbstentionBench is a benchmark for the holistic evaluation of abstention capabilities in frontier LLMs**, spanning **20 datasets** (including 3 new underspecified reasoning challenges) over **6 abstention scenarios** (ranging from underspecified context to stale data).
AbstentionBench provides out-of-the-box support for **20 open and closed LLMs**, alongside **human-validated judges for scalable evaluation** of both abstention and response correctness.

In our accompanying paper, we find that abstention remains a key problem even for frontier LLMs, with model scale having almost no effect. 
Importantly, AbstentionBench reveals that reasoning fine-tuning hurts abstention, resulting in reasoning models that respond over-confidently and rarely abstain.

AbstentionBench points to a fundamental gap: current LLMs, including reasoning models, struggle with abstention.
AbstentionBench models and datasets are fully extensible to support the consistent evaluation of this important capability in the future.

## Explore AbstentionBench Results

You can explore existing abstention results without special installations. Simply download the csv of results and explore away:

```python
import pandas as pd

df = pd.read_csv("analysis/abstention_performance.csv")
df
```

![alt text](sample_results.png)


## Installation

### First-time setup

<details > 
<summary>[Pre-requisite] Before you start, make sure you have mamba installed.
</summary>
  
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && bash Miniforge3-$(uname)-$(uname -m).sh
```
See [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html#fresh-install-recommended) for more instructions.
</details>

<br/>

To setup your environment, 1) clone the repo then 2) run the setup script:
```
source setup.sh
```
<br/>

<details > 
<summary>[Optional] Manual installation
</summary>

1. Create a new environment: `mamba env create -f environment.yml`
2. Install vLLM: [install vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
3. Install PyTorch 
4. `pip install -e .`

If using a GPU, please confirm PyTorch is installed with a compatible CUDA version.
</details>

<details > 
<summary>[Optional] If you want to use GPT4 API:
</summary>
From the repo root:

```
mkdir .secrets
echo YOUR-API-KEY-HERE > .secrets/azure_api_key
```
</details>


## A note about datasets

All datasets required for AbstentionBench should automatically download when running a sweep, with one exception, FreshQA, because it requires exporting from the FreshQA-maintained Google Sheet. 

<details > 
<summary>Setup FreshQA
</summary>

By default, FreshQA runs will fail due to missing data.

To set up to run with FreshQA:

1. Review the list of available FreshQA Google Sheets on the project's [README](https://github.com/freshllms/freshqa?tab=readme-ov-file)
2. Pick a 'baseline' date, which should be before the model's pretraining cut-off, and an 'updated' date, which should be after the cut-off
3. For both baseline and updated, open the Google Sheet and export their contents as CSV (File > Download > Comma Separated Values)
4. Move these two CSVs into `data/freshqa/`, and update the paths in `configs/dataset/freshqa.yaml` accordingly

Note: To exactly replicate our work, use [FreshQA_v10282024](https://docs.google.com/spreadsheets/d/1j6qr14l8oK_7gJ_XdnTBi8Pj1NVt5yeQEBKxFPkKn_g/edit?gid=334049794#gid=334049794) and [FreshQA_v12182024](https://docs.google.com/spreadsheets/d/1llFQDYuwX95L7yYDQ4aLCwJmkEh9VOSHNu6g7HjT8e0/edit?gid=334049794#gid=334049794) as baseline and updated respectively.

</details>

## Run experiments

Activate the environment:
```
source activate.sh
```

For a simple test of the end-to-end pipeline on your local machine, run:
```
python main.py -m mode=local model=dummy abstention_detector=contains_abstention_keyword run_single_job_for_inference_and_judge=True
```

Where the `dataset` arg corresponds to config file names in `configs/dataset/`, and `model` to `config/model/`. See below for supported models and datasets.

By default, `main.py` runs the whole pipeline from inference, through abstention detection, to evaluation. The pipeline launches seaparate jobs for inference and the rest of the evaluations to support a separate LLM judge. To launch everything in a single job for debugging use the `run_single_job_for_inference_and_judge=True` flag.

Model responses and evaluation results will be saved in `save_dir` specified in config.


To run a fast subset on with at most 100 examples per dataset: 
```
python main.py -m mode=cluster dataset='glob(*,exclude=dummy)' model=llama_3_1_8B_instruct sweep_folder=fast-subset dataset_indices_path=data/fast-subset-indices.json
```

Note `mode=cluster` expects you're running the code on a cluster that supports SLURM.

### Running larger models
- Both Llama3.1 70B and Llama3.1 405B FP8 require 8 GPUs (1 node) for inference. Llama3.1 70B can run on V100 GPUs while Llama3.1 405B FP8 requires A100 GPUs.
- Due to vLLM limitations, it is currently not possible to run inference and LLM judge together if they require different amount of resources. Thus, we run Llama3.1 70B and Llama3.1 405B FP8 with `only_run_inference=True`.
- You can launch with Llama3.1 70B with `python main.py -m mode=cluster_one_node_v100 model=llama_3_1_70B_instruct only_run_inference=True` and Llama3.1 405B FP8 with `python main.py -m mode=cluster_one_node_a100 model=llama_3_1_405B_instruct only_run_inference=True`.



### Available models
* DeepSeek-R1 distilled into Llama 70B: `deepseek_r1_distill_llama_70B`
* GPT-4o: `gpt4o`
* Llama 3.1 8B Base: `llama_3_1_8B_base`
* Llama 3.1 8B Instruct: `llama_3_1_8B_instruct`
* Llama 3.1 70B Base: `llama_3_1_70B_base`
* Llama 3.1 70B Instruct: `llama_3_1_70B_instruct`
* Llama 3.1 405B Instruct FP8: `llama_3_1_405B_instruct`
* Mistral 78 Instruct (v0.3): `mistral_7B_instruct_v0_3`
* OLMo 7B Instruct (0724): `olmo_7B_0724_instruct`
* S1 v1 32B: `s1_1_32B`
* Qwen 2.5 32B Instruct: `qwen2_5_32B_instruct`

### Available datasets
* ALCUNA [subsampled]: `alcuna`
* Averitec [subsampled]: `averitec`
* Bias Benchmark for QA (BBQ) [subsampled]: `bbq`
* BIG-Bench Disambiguate: `big_bench_disambiguate`
* BIG-Bench Known Unknowns: `big_bench_known_unknowns`
* CoCoNot: `coconot`
* FalseQA: `falseqa`
* FreshQA: `freshqa`
* Known Unknown Questions (KUQ) [subsampled]:`kuq`
* MediQ: `mediq`
* MoralChoice: `moralchoice`
* Musique: `MusiqueDataset`
* NaturalQuestions [subsampled]: `NQ`
* Question Answering with Questionable Assumptions (QAQA): `qaqa`
* QASPER: `qasper`
* SelfAware: `self_aware`
* SituatedQA: `situatedqa`
* SQuAD 2.0 [subsampled]: `squad2`
* WorldSense: `worldsense`

Note that all datasets with over a threshold 3.5k samples were truncated using uniform subsampling.
These datasets are marked with "[subsampled]" in the list above. 
Sample indices can be found in `data/subsampling-indices.json`. 
You can disable this by setting the `dataset_indices_path` config attribute to `null`. 


### Producing evaluation metrics

To produce F1 Score table, 

```python
from analysis.load_results import Results
from analysis.tables import AbstentionF1ScoreTable

results = Results()
abstention_f1_score = AbstentionF1ScoreTable(results=results)
print(abstention_f1_score.table_df)
```

<img width="626" alt="image" src="https://github.com/user-attachments/assets/6e7f6a35-098c-41cf-bf71-3cf15f2bd2fd" />


## Testing
To run all tests:
```
python -m pytest tests
```

To run just a single file:
```
python -m pytest tests/test_datasets.py
```

By default tests with @slow and @api are skipped, to run all test including slow ones and those with remote API calls:

```
python -m pytest tests --runslow --runapi
```


# Contributing

Please see contributing.md 

# License

Please see license file.

Please note:
Third party content pulled from other locations are subject to its own licenses and you may have other legal obligations or restrictions that govern your use of that content.

# Citation

```
@misc{kirichenko2025abstentionbenchreasoningllmsfail,
      title={AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions}, 
      author={Polina Kirichenko and Mark Ibrahim and Kamalika Chaudhuri and Samuel J. Bell},
      year={2025},
      eprint={2506.09038},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.09038}, 
}
```

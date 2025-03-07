<!-- PROJECT LOGO -->
<!-- <br />
<div align="center">
  <a href="https://github.com/Leo-Lsc/MM-Detect">
    <img src="" alt="Logo" width="120" height="120">
  </a>

  <h3 align="center">MM-Detect</h3>

  <p align="center">
    The First Multimodal Data Contamination Detection Framework
  </p>
</div> -->

# 🕵️ MM-Detect: The First Multimodal Data Contamination Detection Framework
**Dingjie Song**, **Sicheng Lai**, Shunian Chen, Lichao Sun, Benyou Wang*

[🤗 Paper](https://huggingface.co/papers/2411.03823#community) | [📖 arXiv](https://arxiv.org/abs/2411.03823)

<!-- TABLE OF CONTENTS
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#overview">Overview</a>
    </li>
    <li>
      <a href="#run-mm-detect">Run MM-Detect</a>
      <ul>
        <li><a href="#pip-installation">Pip-Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
  </ol>
</details> -->

<!-- ABOUT THE PROJECT -->

<!-- # 🌈 Update -->

## Overview

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

The rapid progression of multimodal large language models (MLLMs) has demonstrated superior performance on various multimodal benchmarks. However, the issue of data contamination during training creates challenges in performance evaluation and comparison. While numerous methods exist for detecting dataset contamination in large language models (LLMs), they are less effective for **MLLMs** due to their various modalities and multiple training phases. Therefore, we introduce a multimodal data contamination detection framework, **MM-Detect**. Besides, we employ a heuristic method to discern whether the contamination originates from the **pre-training phase of LLMs**.

<!-- <div align="center">
  <img src="images/figure1.png" alt="MM-Detect">
</div> -->

<p align="center">
  <img src="images/figure1.png" alt="MM-Detect">
</p>

## 🤖 Environment Setup
```bash
git clone https://github.com/FreedomIntelligence/MM-Detect.git
conda create -n MM-Detect python=3.10
cd MM-Detect
pip install torch==2.1.2
pip install -r requirements.txt
pip install googletrans==3.1.0a0
pip install httpx==0.27.2
```

Ensure that your system has Java installed to enable the use of the [Stanford POS Tagger](https://nlp.stanford.edu/).

```bash
sudo apt update
sudo apt install openjdk-11-jdk
```

<!-- GETTING STARTED -->
## 🚀 Run MM-Detect
Our codebase supports the following models on [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar), [COCO-Caption](https://huggingface.co/datasets/lmms-lab/COCO-Caption2017), [Nocaps](https://huggingface.co/datasets/lmms-lab/NoCaps) and [Vintage](https://huggingface.co/datasets/SilentAntagonist/vintage-artworks-60k-captioned):

- **White-box Models:**
  - `LLaVA-1.5`
  - `VILA1.5`
  - `Qwen-VL-Chat`
  - `idefics2`
  - `Phi-3-vision-instruct`
  - `Yi-VL`
  - `InternVL2`
  - `DeepSeek-VL2`

- **Grey-box Models:**
  - `fuyu`

- **Black-box Models:**
  - `GPT-4o`
  - `Gemini-1.5-Pro`
  - `Claude-3.5-Sonnet`

🔐 **Important**: When detecting contamination of black-box models, ensure to add your API key at `Line 26` in `mm_detect/mllms/gpt.py`:
```bash
api_key='your-api-key'
```

🌱 To **save intermediate results** and enable the **Resume** function, please add your output_dir at `line 77` in `multimodal_methods/option_order_sensitivity_test.py` and at `line 104` in `multimodal_methods/slot_guessing_for_perturbation_caption.py`:
```bash
results_file = "output_dir/results.json"
```

📌 To run contamination detection for MLLMs, you can follow the multiple test scripts in `scripts/tests/mllms` folder. For instance, use the following command to run **Option Order Sensitivity Test** on ScienceQA with GPT-4o:
```bash
bash scripts/mllms/option_order_sensitivity_test/test_ScienceQA.sh -m gpt-4o
```
## 🔍 Discern the Source of Contamination
We support the following LLMs on [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar):

- **LLMs:**
  - `LLaMA2`
  - `Qwen`
  - `Internlm2`
  - `Mistral`
  - `Phi-3-instruct`
  - `Yi`
  - `DeepSeek-MoE-Chat`

📌 For instance, use the following command to run the Qwen-7B:
``` bash
bash scripts/llms/detect_pretrain/test_MMStar.sh -m Qwen/Qwen-7B
```

## 🎉 Citation

⭐ If you find our implementation and paper helpful, please consider citing our work and starring the repository⭐:
```bibtex
@misc{song2024textimagesleakedsystematic,
  title={Both Text and Images Leaked! A Systematic Analysis of Multimodal LLM Data Contamination},
  author={Dingjie Song and Sicheng Lai and Shunian Chen and Lichao Sun and Benyou Wang},
  year={2024},
  eprint={2411.03823},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.03823},
}
```

## 🛠 Troubleshooting

If you encounter the following error when using [googletrans](https://pypi.org/project/googletrans/):

```python
AttributeError: module 'httpcore' has no attribute 'SyncHTTPTransport'
```
please refer to the solution provided on [this Stack Overflow page](https://stackoverflow.com/questions/72796594/attributeerror-module-httpcore-has-no-attribute-synchttptransport) for further guidance.

## ❤️ Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [LLMSanitize](https://github.com/ntunlp/LLMSanitize)
- [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml)

<!-- Contributors -->
<!-- ## Contributors
- [Leo-Lsc](https://github.com/Leo-Lsc)
- [bbsngg](https://github.com/bbsngg) -->

<!-- <table>
  <tr>
    <td align="center">
      <a href="https://github.com/Leo-Lsc">
        <img src="https://avatars.githubusercontent.com/u/124846947?v=4" width="50" height="50" style="border-radius: 50%; overflow: hidden;" alt="Leo-Lsc"/>
        <br />
        <sub><b>Leo-Lsc</b></sub>
      </a>
    </td>
  </tr>
</table> -->

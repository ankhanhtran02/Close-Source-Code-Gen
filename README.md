# Close Source Code Generation

## Overview

This repository provides a script for generating code completions using the Gemini API and a HuggingFace dataset. It is designed to automate the process of sending prompts to the Gemini model and saving the generated responses for further analysis or evaluation. The script supports configurable parameters such as model selection, temperature, top-p, top-k, candidate count, and output token limits.

## Instructions

### 1. Install Dependencies
Run the following commands in your terminal:
```sh
conda create -n close-source python=3.11
conda activate close-source
pip install -r requirements.txt
```
### 2. Generate code completions
Get your [Gemini API key](https://aistudio.google.com/apikey) then run the following command in your terminal:

```sh
python3 generate.py \
  --api_key YOUR_API_KEY \
  --data AnhMinhLe/repoexec_comparison_dataset \
  --split repoexec_bm25_final \
  --save_path outputs/repoexec_bm25_final_generated.jsonl \
```

View `generate.py` module for descriptions of arguments. Use `--continue_last_generation` argument to resume the last generation.


# Puisi & Pantun Generator

Forked from [IlhamFP](https://github.com/ilhamfp/puisi-pantun-generator)

## Quickstart

1. Create a new virtual environment
2. Install everything in `requirements.txt`
3. Run this command: `python run_generation.py --model_type gpt2 --model_name_or_path gpt2 --prompt <INSERT YOUR PROMPT HERE>`

### Notes on Quickstart

- There are other `--model_type` besides `gpt2`, just search for `MODEL_CLASSES` in `run_generation.py` file.
- If you do not provide a prompt in the command, meaning you just run this command: `python run_generation.py --model_type gpt2 --model_name_or_path gpt2` instead of the command above, then the program will ask for a prompt when you run it.
- At the time of writing (23rd April 2024), the program does not produce any pantun.

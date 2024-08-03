# Puisi & Pantun Generator

Forked from [IlhamFP](https://github.com/ilhamfp/puisi-pantun-generator)

## Quickstart

1. Create a new virtual environment
2. Install everything in `requirements.txt`. For example: `pip install -r requirements.txt`
3. Try running this example command: `python run_generation.py --model_type gpt2 --model_name_or_path gpt2 --prompt sumur`
4. Expect to see gibberish (proof that you are smarter than computers).
5. Run `python run_language_modeling.py --model_type gpt2 --model_name_or_path gpt2 --output_dir output/`
6. Try running the example command again: `python run_generation.py --model_type gpt2 --model_name_or_path gpt2 --prompt sumur`
7. You should see some legit words in Indonesian. If not, please refer to the [Troubleshooting](github.com/irfanzainudin/puisi-pantun-generator) section.

## Important information

Run this command to language model: `python run_language_modeling.py --model_type <INSERT YOUR MODEL HERE> --model_name_or_path <INSERT YOUR MODEL HERE> --output_dir output/`

Run this command to generate: `python run_generation.py --model_type <INSERT YOUR MODEL HERE> --model_name_or_path <INSERT YOUR MODEL HERE> --prompt <INSERT YOUR PROMPT HERE>`

Available models:

```
MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}
```

### Notes on Quickstart

- If you do not provide a prompt in the command, meaning you just run this command: `python run_generation.py --model_type gpt2 --model_name_or_path gpt2` instead of the command above, then the program will ask for a prompt when you run it.
- At the time of writing (23rd April 2024), the program does not produce any pantun.

### Full usage options

For `run_generation.py`:

```
usage: run_generation.py [-h] --model_type MODEL_TYPE --model_name_or_path
                         MODEL_NAME_OR_PATH [--prompt PROMPT]
                         [--length LENGTH] [--stop_token STOP_TOKEN]
                         [--temperature TEMPERATURE]
                         [--repetition_penalty REPETITION_PENALTY] [--k K]
                         [--p P] [--padding_text PADDING_TEXT]
                         [--xlm_language XLM_LANGUAGE] [--seed SEED]
                         [--no_cuda]
                         [--num_return_sequences NUM_RETURN_SEQUENCES]
```

For `run_language_modeling.py`:

```
usage: run_language_modeling.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH] [--model_type MODEL_TYPE] [--config_name CONFIG_NAME]
                                [--tokenizer_name TOKENIZER_NAME] [--cache_dir CACHE_DIR] [--train_data_file TRAIN_DATA_FILE]
                                [--eval_data_file EVAL_DATA_FILE] [--line_by_line [LINE_BY_LINE]] [--mlm [MLM]] [--mlm_probability MLM_PROBABILITY]
                                [--block_size BLOCK_SIZE] [--overwrite_cache [OVERWRITE_CACHE]] --output_dir OUTPUT_DIR
                                [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]] [--do_train [DO_TRAIN]] [--do_eval [DO_EVAL]] [--do_predict [DO_PREDICT]]
                                [--evaluation_strategy {no,steps,epoch}] [--prediction_loss_only [PREDICTION_LOSS_ONLY]]
                                [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE] [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
                                [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE] [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--eval_accumulation_steps EVAL_ACCUMULATION_STEPS]
                                [--eval_delay EVAL_DELAY] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--adam_beta1 ADAM_BETA1]
                                [--adam_beta2 ADAM_BETA2] [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM]
                                [--num_train_epochs NUM_TRAIN_EPOCHS] [--max_steps MAX_STEPS]
                                [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau}]
                                [--warmup_ratio WARMUP_RATIO] [--warmup_steps WARMUP_STEPS] [--log_level {debug,info,warning,error,critical,passive}]
                                [--log_level_replica {debug,info,warning,error,critical,passive}] [--log_on_each_node [LOG_ON_EACH_NODE]]
                                [--no_log_on_each_node] [--logging_dir LOGGING_DIR] [--logging_strategy {no,steps,epoch}]
                                [--logging_first_step [LOGGING_FIRST_STEP]] [--logging_steps LOGGING_STEPS]
                                [--logging_nan_inf_filter [LOGGING_NAN_INF_FILTER]] [--no_logging_nan_inf_filter] [--save_strategy {no,steps,epoch}]
                                [--save_steps SAVE_STEPS] [--save_total_limit SAVE_TOTAL_LIMIT] [--save_safetensors [SAVE_SAFETENSORS]]
                                [--save_on_each_node [SAVE_ON_EACH_NODE]] [--no_cuda [NO_CUDA]] [--use_mps_device [USE_MPS_DEVICE]] [--seed SEED]
                                [--data_seed DATA_SEED] [--jit_mode_eval [JIT_MODE_EVAL]] [--use_ipex [USE_IPEX]] [--bf16 [BF16]] [--fp16 [FP16]]
                                [--fp16_opt_level FP16_OPT_LEVEL] [--half_precision_backend {auto,cuda_amp,apex,cpu_amp}]
                                [--bf16_full_eval [BF16_FULL_EVAL]] [--fp16_full_eval [FP16_FULL_EVAL]] [--tf32 TF32] [--local_rank LOCAL_RANK]
                                [--ddp_backend {nccl,gloo,mpi,ccl}] [--tpu_num_cores TPU_NUM_CORES] [--tpu_metrics_debug [TPU_METRICS_DEBUG]]
                                [--debug DEBUG] [--dataloader_drop_last [DATALOADER_DROP_LAST]] [--eval_steps EVAL_STEPS]
                                [--dataloader_num_workers DATALOADER_NUM_WORKERS] [--past_index PAST_INDEX] [--run_name RUN_NAME]
                                [--disable_tqdm DISABLE_TQDM] [--remove_unused_columns [REMOVE_UNUSED_COLUMNS]] [--no_remove_unused_columns]
                                [--label_names LABEL_NAMES [LABEL_NAMES ...]] [--load_best_model_at_end [LOAD_BEST_MODEL_AT_END]]
                                [--metric_for_best_model METRIC_FOR_BEST_MODEL] [--greater_is_better GREATER_IS_BETTER]
                                [--ignore_data_skip [IGNORE_DATA_SKIP]] [--sharded_ddp SHARDED_DDP] [--fsdp FSDP]
                                [--fsdp_min_num_params FSDP_MIN_NUM_PARAMS] [--fsdp_config FSDP_CONFIG]
                                [--fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP] [--deepspeed DEEPSPEED]
                                [--label_smoothing_factor LABEL_SMOOTHING_FACTOR]
                                [--optim {adamw_hf,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_apex_fused,adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit}]
                                [--optim_args OPTIM_ARGS] [--adafactor [ADAFACTOR]] [--group_by_length [GROUP_BY_LENGTH]]
                                [--length_column_name LENGTH_COLUMN_NAME] [--report_to REPORT_TO [REPORT_TO ...]]
                                [--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS] [--ddp_bucket_cap_mb DDP_BUCKET_CAP_MB]
                                [--dataloader_pin_memory [DATALOADER_PIN_MEMORY]] [--no_dataloader_pin_memory]
                                [--skip_memory_metrics [SKIP_MEMORY_METRICS]] [--no_skip_memory_metrics]
                                [--use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]] [--push_to_hub [PUSH_TO_HUB]]
                                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--hub_model_id HUB_MODEL_ID]
                                [--hub_strategy {end,every_save,checkpoint,all_checkpoints}] [--hub_token HUB_TOKEN]
                                [--hub_private_repo [HUB_PRIVATE_REPO]] [--gradient_checkpointing [GRADIENT_CHECKPOINTING]]
                                [--include_inputs_for_metrics [INCLUDE_INPUTS_FOR_METRICS]] [--fp16_backend {auto,cuda_amp,apex,cpu_amp}]
                                [--push_to_hub_model_id PUSH_TO_HUB_MODEL_ID] [--push_to_hub_organization PUSH_TO_HUB_ORGANIZATION]
                                [--push_to_hub_token PUSH_TO_HUB_TOKEN] [--mp_parameters MP_PARAMETERS] [--auto_find_batch_size [AUTO_FIND_BATCH_SIZE]]
                                [--full_determinism [FULL_DETERMINISM]] [--torchdynamo TORCHDYNAMO] [--ray_scope RAY_SCOPE] [--ddp_timeout DDP_TIMEOUT]
                                [--torch_compile [TORCH_COMPILE]] [--torch_compile_backend TORCH_COMPILE_BACKEND]
                                [--torch_compile_mode TORCH_COMPILE_MODE] [--xpu_backend {mpi,ccl,gloo}]
```
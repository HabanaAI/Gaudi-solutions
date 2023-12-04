# Copyright(C) 2023 Habana Labs, Ltd. an Intel Company
"""Inference for FastChat models."""
import abc
from typing import Optional
import warnings

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from fastchat.serve.modeling_llama_hpu import LlamaForCausalLM
from fastchat.serve.modeling_gpt_neox_hpu import GPTNeoXForCausalLM

from fastchat.conversation import conv_templates, get_default_conv_template, SeparatorStyle
from fastchat.serve.compression import compress_module
from fastchat.serve.monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations
from fastchat.serve.serve_chatglm import chatglm_generate_stream
import time
import numpy as np
import os

tps_message = ''
token_count = 0

def raise_warning_for_old_weights(model_path, model):
    if "vicuna" in model_path.lower():
        try:
            is_vicuna = isinstance(model, LlamaForCausalLM)
        except Exception:
            is_vicuna = isinstance(model, LLamaForCausalLM)
        if is_vicuna and model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fschat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.1: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template conv_one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommonded).\n")


def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = torch.cuda.device_count() if max_gpus is None else min(max_gpus, torch.cuda.device_count())

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def compute_skip_echo_len(conv_template, model_name, conv, prompt):
    model_name = model_name.lower()
    if conv_template == 'llama2':
        skip_echo_len = len(prompt) + 1 - prompt.count("</s><s>") * 7
    elif "chatglm" in model_name:
        skip_echo_len = len(conv.messages[-2][1]) + 1
    elif "dolly" in model_name:
        special_toks = ["### Instruction:", "### Response:", "### End"]
        prompt_tmp = prompt
        for tok in special_toks:
            prompt_tmp = prompt_tmp.replace(tok, "")
        skip_echo_len = len(prompt_tmp)
    else:
        skip_echo_len = len(prompt) + 1 - prompt.count("</s>") * 3
    return skip_echo_len


def load_model(model_path, device, num_gpus, max_gpu_memory=None,
               load_8bit=False, debug=False, use_graphs=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.bfloat16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {i: str(int(available_gpu_memory[i] * 0.85)) +
                        "GiB" for i in range(num_gpus)}
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        print("init_kwargs", kwargs)
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    elif device == "hpu":
        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch.hpu.graphs as htgraphs
        import habana_frameworks.torch as ht
        kwargs = {"torch_dtype": torch.bfloat16}
    else:
        raise ValueError(f"Invalid device: {device}")

    config = AutoConfig.from_pretrained(model_path)
    if "chatglm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(device)
    elif "dolly" in model_path:
        kwargs.update({"torch_dtype": torch.bfloat16})
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
        model = GPTNeoXForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        model.config.pad_token_id = model.config.eos_token_id
    elif config.model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path,
            low_cpu_mem_usage=True, **kwargs)
        raise_warning_for_old_weights(model_path, model)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)
    elif device == "hpu":
        if config.model_type not in ['llama', 'gpt_neox']:
            raise NotImplementedError
        model.to(device)
        model = model.eval()
        if use_graphs:
            model = ht.hpu.wrap_in_hpu_graph(model)

    if debug:
        print(model)

    return model, tokenizer

def get_device(model):
    if hasattr(model, 'device'):
        return model.device
    if hasattr(model, 'module'):
        return model.module.device
    assert False, 'Cannot extract device!'
    return None

def prepare_input(model, max_length, static_shapes=False, **model_args):
    device = get_device(model)
    if static_shapes:
        input_ids = model_args['input_ids']
        attention_mask = model_args['attention_mask']
        cur_length = input_ids.shape[-1]
        padding_length = max_length - cur_length
        model_args['token_idx'] = torch.tensor(cur_length)
        model_args['input_ids'] = F.pad(input_ids, (0, padding_length), value=model.config.pad_token_id)
        model_args['attention_mask'] = F.pad(attention_mask, (0, padding_length), value=0)
    if 'token_idx' in model_args:
        model_args['token_idx'] = model_args['token_idx'].to(device)
    model_args['input_ids'] = model_args['input_ids'].to(device)
    model_args['attention_mask'] = model_args['attention_mask'].to(device)
    model_args['use_cache'] = model_args.get('use_cache')
    return model_args


@torch.inference_mode()
def generate_stream(model, tokenizer, params, device,
                    context_len=2048, stream_interval=2,
                    use_cache=False, static_shapes=False,
                    output_tps=False, seed=1):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    if stop_str == tokenizer.eos_token:
        stop_str = None
    max_length=context_len
    model_args = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    model_args = model_args.to(device)
    input_ids = model_args.input_ids
    output_ids = input_ids.tolist()[0]

    global token_count
    token_count = len(output_ids)

    eos_generated = torch.zeros((input_ids.shape[-2],), dtype=torch.bool, device=input_ids.device)

    model_args = prepare_input(model, max_length=max_length, static_shapes=static_shapes, **model_args)
    model_args['use_cache'] = use_cache


    if device == 'hpu':
        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch.hpu.random as htrandom
        htrandom.manual_seed(seed)
        htcore.mark_step()
        g = None
    else:
        g = torch.Generator(device=device).manual_seed(seed)

    t1 = time.time()
    n = 0

    for i in range(max_new_tokens):
        n+=1
        out = model(**model_args)
        logits = out.logits
        past_key_values = out.past_key_values
        if static_shapes and (i == 0 or not use_cache):
            last_token_logits = logits[:,model_args['token_idx'] - 1,:]
        else:
            last_token_logits = logits[:,-1,:]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1, generator=g))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
            yield output

        if stopped:
            break

        next_tokens = torch.logical_not(eos_generated) * token + eos_generated * model.config.pad_token_id
        eos_generated.logical_or_(next_tokens.eq(model.config.eos_token_id))
        next_tokens = next_tokens.unsqueeze(-1)

        # set input_ids for next forward pass
        if use_cache:
            model_args['input_ids'] = next_tokens
            model_args['past_key_values'] = past_key_values
        elif static_shapes:
            model_args['input_ids'].index_copy_(1, model_args['token_idx'], next_tokens)
        else:
            model_args['input_ids'] = torch.cat([model_args['input_ids'], next_tokens], dim=-1)

        # set attention_mask and token_idx
        if static_shapes:
            model_args['attention_mask'].index_fill_(1, model_args['token_idx'], 1)
            model_args['token_idx'].add_(1)
        else:
            model_args['attention_mask'] = F.pad(model_args['attention_mask'], (0, 1), value=1)

        if device == 'hpu':
            htcore.mark_step()

    t2 = time.time()
    if output_tps:
        total_time = round(t2-t1, 3)
        global tps_message
        tps_message = '-------------------------------------------\nTime: ' + str(total_time) + "\tTokens: " + str(n) + '\tTPS: ' + str(round(n/total_time, 2)) + '\n-------------------------------------------'

    token_count += n

    del past_key_values, model_args, out, logits, last_token_logits


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream, skip_echo_len: int):
        """Stream output."""


def chat_loop(model_path: str, device: str, num_gpus: str,
              max_gpu_memory: str, load_8bit: bool,
              conv_template: Optional[str], temperature: float,
              max_new_tokens: int, chatio: ChatIO,
              debug: bool, use_graphs: bool, use_cache: bool,
              static_shapes: bool, context_len: int, output_tps: bool,
              seed: int):
    # Model
    model, tokenizer = load_model(model_path, device,
        num_gpus, max_gpu_memory, load_8bit, debug, use_graphs)
    is_chatglm = "chatglm" in str(type(model)).lower()

    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = get_default_conv_template(model_path).copy()

    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        if inp == 'clear':
            conv = conv_templates[conv_template].copy() if conv_template else get_default_conv_template(model_path).copy()
            os.system('cls' if os.name == 'nt' else 'clear')
            continue

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            prompt = conv.messages[conv.offset:]
            generate_stream_func = chatglm_generate_stream
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

        skip_echo_len = compute_skip_echo_len(conv_template, model_path, conv, prompt)

        params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, params, device,
                                             use_cache=use_cache, static_shapes=static_shapes,
                                             context_len=context_len, output_tps=output_tps,
                                             seed=seed)
        outputs = chatio.stream_output(output_stream, skip_echo_len)
        if output_tps:
            print(tps_message)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
            print('Token count: ' + str(token_count))

        if token_count + max_new_tokens > context_len:
            print('\n\nThe session history has been cleared because the maximum context length was reached\n\n')
            if conv_template:
                conv = conv_templates[conv_template].copy()
            else:
                conv = get_default_conv_template(model_path).copy()

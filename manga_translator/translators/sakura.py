from typing import List
import os
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

from .common import OfflineTranslator

# Adapted from:
# https://github.com/SakuraLLM/Sakura-13B-Galgame


class SakuraTranslator(OfflineTranslator):
    """Sakura translator based on Sakura13B large language model.
    Now the inference backend has been implemented contain:
    a) llama.cpp (with metal support)
    """

    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_SUB_DIR, "sakura")
    _MODEL_MAPPING = {
        "model": {
            "url": "https://huggingface.co/SakuraLLM/Sakura-13B-LNovel-v0.8-GGUF/resolve/main/sakura-13b-lnovel-norm-Q5_K_M.gguf",
            "hash": "a9f503830ac1ffdab9cd44d16b0470f19bd2b269e806b59818e9c2f72dfa4f83",
        },
    }
    _MODEL_VERSION = "0.8"
    _SAKURA_MODEL_NAME = 'sakura-13b-lnovel-norm-Q5_K_M.gguf'
    _TEXT_LENGTH = 256

    async def _load(self, from_lang: str, to_lang: str, device: str):
        self.load_params = {
            "from_lang": from_lang,
            "to_lang": to_lang,
            "device": device,
        }
        print('Llama: ', self._get_file_path())
        n_gpu_layers = -1 if device == "mps" else 0
        self.model = Llama(
            model_path=self._get_file_path(self._SAKURA_MODEL_NAME),
            n_gpu_layers=n_gpu_layers,
            n_ctx=4 * self._TEXT_LENGTH,
            verbose=False,
        )

    async def _unload(self):
        del self.model

    async def _infer(
        self, from_lang: str, to_lang: str, queries: List[str]
    ) -> List[str]:
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.3,
            top_k=40,
            num_beams=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=512,
            min_new_tokens=1,
            do_sample=True,
        )

        tokenizer = None

        data = []
        for d in tqdm(queries):
            prompt = self.get_prompt(d, self._MODEL_VERSION)
            output = self.get_model_response(
                self.model,
                tokenizer,
                prompt,
                self._MODEL_VERSION,
                generation_config,
                self._TEXT_LENGTH,
                True,
            )
            output.strip()
            # trancated chaos reput
            output = output[: 2 * len(d)]
            data.append(output.strip())

        return data

    def get_prompt(self, input, model_version="0.8"):
        if model_version == "0.5" or model_version == "0.8":
            prompt = "<reserved_106>将下面的日文文本翻译成中文：" + input + "<reserved_107>"
            return prompt
        raise ValueError()

    def get_model_response(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        model_version: str,
        generation_config: GenerationConfig,
        text_length: int,
        llama_cpp: bool = True,
        use_llm_sharp: bool = False,
    ):
        backup_generation_config_stage2 = GenerationConfig(
            temperature=0.1,
            top_p=0.3,
            top_k=40,
            num_beams=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=text_length,
            min_new_tokens=1,
            do_sample=True,
            repetition_penalty=1.0,
            frequency_penalty=0.05,
        )

        backup_generation_config_stage3 = GenerationConfig(
            temperature=0.1,
            top_p=0.3,
            top_k=40,
            num_beams=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_new_tokens=text_length,
            min_new_tokens=1,
            do_sample=True,
            repetition_penalty=1.0,
            frequency_penalty=0.2,
        )

        backup_generation_config = [
            backup_generation_config_stage2,
            backup_generation_config_stage3,
        ]

        def generate(model, generation_config):
            if "frequency_penalty" in generation_config.__dict__.keys():
                output = model(
                    prompt,
                    max_tokens=generation_config.__dict__["max_new_tokens"],
                    temperature=generation_config.__dict__["temperature"],
                    top_p=generation_config.__dict__["top_p"],
                    repeat_penalty=generation_config.__dict__["repetition_penalty"],
                    frequency_penalty=generation_config.__dict__["frequency_penalty"],
                )
            else:
                output = model(
                    prompt,
                    max_tokens=generation_config.__dict__["max_new_tokens"],
                    temperature=generation_config.__dict__["temperature"],
                    top_p=generation_config.__dict__["top_p"],
                    repeat_penalty=generation_config.__dict__["repetition_penalty"],
                )
            return output

        stage = 0
        output = generate(model, generation_config)
        while output["usage"]["completion_tokens"] == text_length:
            stage += 1
            if stage > 2:
                print("model degeneration cannot be avoided.")
                break
            print("model degeneration detected, retrying...")
            output = generate(model, backup_generation_config[stage - 1])
        response = output["choices"][0]["text"]
        return response

    def supports_languages(
        self, from_lang: str, to_lang: str, fatal: bool = False
    ) -> bool:
        return from_lang == 'JPN' and to_lang == 'CHS'

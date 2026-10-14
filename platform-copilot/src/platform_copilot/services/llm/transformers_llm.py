"""Local text-generation LLM via Hugging Face Transformers.

Used by the self-contained Hugging Face Space demo, where no external inference
service is available: it loads a small instruct model and runs it on CPU, so a single
container serves the whole app with no API key or network at request time. Generation
is slow compared with a served model — fine for a demo, not for production (the real
service uses Ollama / a served endpoint). Implements the LLM Protocol.

Needs the `transformers` extra:  pip install ".[transformers]"
"""

from __future__ import annotations

import threading


class TransformersLLM:
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        *,
        max_new_tokens: int = 320,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32)
        self._model.eval()
        self._max_new_tokens = max_new_tokens
        # A single model instance is not safe to call concurrently; the Gradio
        # Space can dispatch overlapping requests, so serialize generation.
        self._lock = threading.Lock()

    def generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with self._lock, self._torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

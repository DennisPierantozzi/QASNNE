from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,       # e.g., google/medgemma-4b-it
    MllamaForConditionalGeneration,    # meta-llama/Llama-3.2-11B-Vision-Instruct
    Qwen2_5_VLForConditionalGeneration # Qwen/Qwen2.5-VL-*
)
from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from huggingface_hub import login
from dataclasses import dataclass
from typing import Optional
import torch
from PIL import Image
from transformers import GPT2Tokenizer
from peft import LoraConfig, TaskType
from model import PitVQALora, SurgicalGPTGen
from .peft_utils import batch_greedy_search, generate_sampled_answer, generate_sampled_answer_surgical
import numpy as np
from typing import Tuple, List, Dict

def format_chat(system_prompt: str, pil_image: Image.Image, question: str):
    return [
        {"role":"system","content":[{"type":"text","text":system_prompt}]},
        {"role":"user","content":[{"type":"image","image":pil_image},{"type":"text","text":question}]},
    ]

# ==============================
# Class-based VLM unification
# ==============================
class VLMClient:
    """
    Unified loader/runner for HF VLMs and PitVQALora.
    Holds model_id, model, and processor/tokenizer, and exposes `answer(...)`.
    """

    def __init__(self, model_id: str, device: str = "cuda", is_peft: bool = False):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None   # HF processors OR GPT2Tokenizer for pitvqa
        self._is_pit = False
        self._is_surg = False
        self._load()

    # ---------- Public API ----------
    def vlm_answer(
        self,
        pil_image,
        question: str,
        *,
        device: str = "cuda",
        temperature: float = 1.0,
        max_new_tokens: int = 100,
        top_p: float = 0.9,
        top_k: int = 50,
        system_prompt: str = "You are a helpful vision-language assistant.",
        sampled: bool = False,
        seed: Optional[int] = None,
        num_samples: int = 1,
    ) -> Tuple[str, Dict[str, float]]:
        

        ## Special handling for PitVQALora and SurgicalGPT
        
        if self._is_pit or self._is_surg:

            # Custom generation for PitVQALora
            # Convert PIL Image to tensor for PitVQA
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Adjust size as needed
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(pil_image).unsqueeze(0).to(device)

            if not sampled:
                # batch_greedy_search expects questions as a list
                answers, token_entropies_dict = batch_greedy_search(
                    image_tensor, [question], self.model, self.processor, max_new_tokens, device=device, model_type="surgical" if self._is_surg else "pitlora", compute_metrics=True
                )
                
                # Extract single result from batch (index 0)
                answer = answers[0]
                metrics = {
                    "avg_prob": token_entropies_dict["avg_prob"][0],
                    "max_prob": token_entropies_dict["max_prob"][0],
                    "avg_entropy": token_entropies_dict["avg_entropy"][0],
                    "max_entropy": token_entropies_dict["max_entropy"][0],
                }
                
                return answer, metrics
            
            else:

                answers, _ = generate_sampled_answer(
                    image_tensor, [question], self.model, self.processor, max_new_tokens, device,
                    temperature, top_k, top_p, seed=seed) if self._is_pit else generate_sampled_answer_surgical(
                    image_tensor, [question], self.model, self.processor, max_new_tokens, device,
                    temperature, top_k, top_p, seed=seed)
                
                return answers[0] if answers else ""

        ## Standard handling for HF VLMs

        # Build inputs
        chat_messages = format_chat(system_prompt, pil_image, question)
        text_in = self.processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_in], images=pil_image, return_tensors="pt").to(device)

        in_len = inputs.input_ids.shape[1]

        # Ensure pad token exists
        if getattr(self.processor, "tokenizer", None) and self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        gen_cfg = GenerationConfig.from_model_config(self.model.config)
        gen_cfg.do_sample = sampled
        gen_cfg.temperature = 0.1 if not sampled else temperature  # ignored when do_sample=False
        gen_cfg.top_k = top_k
        gen_cfg.top_p = top_p
        gen_cfg.num_return_sequences = 1
        gen_cfg.max_new_tokens = max_new_tokens
        gen_cfg.output_scores = True
        gen_cfg.return_dict_in_generate = True
        
        gen_cfg.num_return_sequences = num_samples if sampled else 1
        if seed is not None:
            torch.manual_seed(seed)

        out = self.model.generate(**inputs, generation_config=gen_cfg)
        out_ids = out.sequences[0][in_len:]     # generated token ids
        scores = out.scores                      # list[T] of logits tensors [1, V]

        # Truncate at first EOS → exclude EOS from metrics
        eos_id = self.model.generation_config.eos_token_id
        if isinstance(eos_id, (list, tuple, set)):
            eos_mask = torch.isin(out_ids, torch.tensor(list(eos_id), device=out_ids.device))
        else:
            eos_mask = (out_ids == eos_id)
        eos_pos = eos_mask.nonzero(as_tuple=True)[0]
        gen_len = eos_pos[0].item() if len(eos_pos) > 0 else out_ids.size(0)

        # Decode text
        txt = self.processor.tokenizer.decode(
            out_ids[:gen_len], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

        if sampled:
            sequences = out.sequences[:, in_len:]  # [num_samples, T]
            answers = self.processor.batch_decode(sequences, skip_special_tokens=True)
            return answers  # list[str]
        # Compute uncertainty metrics

        # Per-step entropy (full-vocab) and NLL of chosen token (stable log-sum-exp)
        token_entropies: List[float] = []
        token_nlls: List[float] = []

        for j in range(gen_len):
            if j >= len(scores):
                break
            logits = scores[j][0].float()  # [V]

            # logZ via log-sum-exp; logP; entropy in nats
            m = torch.max(logits)
            logZ = m + torch.log(torch.exp(logits - m).sum())
            logP = logits - logZ
            H = -(torch.exp(logP) * logP).sum()          # scalar
            token_entropies.append(float(H.item()))

            tok_id = int(out_ids[j].item())
            nll = (logZ - logits[tok_id])                # -log p(token_j)
            token_nlls.append(float(nll.item()))

        if len(token_nlls) == 0:
            metrics = {"avg_prob": 0.0, "max_prob": 0.0, "avg_entropy": 0.0, "max_entropy": 0.0}
        else:
            metrics = {
                "avg_prob": float(np.mean(token_nlls)),       # AvgProb  (mean NLL)
                "max_prob": float(np.max(token_nlls)),        # MaxProb  (max  NLL)
                "avg_entropy": float(np.mean(token_entropies)),  # AvgEnt
                "max_entropy": float(np.max(token_entropies)),   # MaxEnt
            }

        return txt, metrics

    # ---------- Loading ----------
    def _load(self):
        mid = self.model_id

        if mid == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            self.processor = AutoProcessor.from_pretrained(mid)
            self.model = MllamaForConditionalGeneration.from_pretrained(
                mid, torch_dtype=torch.bfloat16, device_map={"": torch.device(self.device)}, low_cpu_mem_usage=False
            ).to(self.device).eval()
            return

        if mid == "Qwen/Qwen2.5-VL-3B-Instruct":
            self.processor = AutoProcessor.from_pretrained(mid)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                mid, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
            ).eval()
            return

        if mid == "google/medgemma-4b-it":
            self.model = AutoModelForImageTextToText.from_pretrained(
                mid, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
            ).eval()
            self.processor = AutoProcessor.from_pretrained(mid)
            return

        if mid == "pitLoRA":
            # Paths/settings (kept identical to your function; adjust if needed)
            vec_weight_path = '/SAN/medic/Cholec/PitVQAGen/best_model.pth'

            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj"]
            )

            model = PitVQALora(peft_config=lora_config)
            checkpoint = torch.load(vec_weight_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            self.model = model.to(self.device).eval()

            # Mark and store tokenizer as "processor"
            self.processor = tokenizer
            self._is_pit = True
            return
        
        if mid == "surgicalGPT":
            # Paths/settings (kept identical to your function; adjust if needed)
            save_dir = '/SAN/medic/Cholec/SurgicalGPTGen/'
            model_name = 'best_model.pth'
            model_path = save_dir + model_name
            print(f'model name: {model_path}')

            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj"]
            )

            model = SurgicalGPTGen(peft_config=lora_config)
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            self.model = model.to(self.device).eval()

            # Mark and store tokenizer as "processor"
            self.processor = tokenizer
            self._is_surg = True
            return

        raise ValueError(f"Unsupported model_id: {mid}")

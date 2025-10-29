import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def batch_greedy_search(
    images, questions, model, tokenizer, max_length, device,
    model_type="pitlora",  # "surgical" or "pitlora"
    compute_metrics=True    # Whether to compute entropy/NLL
):
    """
    Unified greedy decoding that works with both SurgicalGPTGen and PitVQALora.
    
    Args:
        model_type: "surgical" for SurgicalGPTGen, "lora" for PitVQALora
        compute_metrics: If True, compute per-step entropy and NLL
    
    Returns:
        answers: List[str]
        token_entropies: Dict (if compute_metrics=True), else None
    """
    batch_size = len(questions)
    answers = []
    
    if compute_metrics:
        step_entropies = [[] for _ in range(batch_size)]
        step_nlls = [[] for _ in range(batch_size)]
    
    model.eval()
    
    # Build prompts
    prompt_texts = [f"Question: {q}\nAnswer:" for q in questions]
    prompt_inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding="longest",
        add_special_tokens=False,
    )
    
    # Prepare model inputs
    padded_input_ids = torch.full(
        (batch_size, max_length), 
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long, 
        device=device
    )
    padded_attention_mask = torch.zeros((batch_size, max_length), device=device)
    
    orig_length = prompt_inputs["input_ids"].size(1)
    orig_length = min(orig_length, max_length)
    padded_input_ids[:, :orig_length] = prompt_inputs["input_ids"][:, :orig_length].to(device)
    padded_attention_mask[:, :orig_length] = prompt_inputs["attention_mask"][:, :orig_length].to(device)
    
    images = images.to(device)
    only_answer_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    valid_lengths = padded_attention_mask.sum(dim=1).long()
    batch_indices = torch.arange(batch_size, device=device)
    
    # Generation loop
    while True:
        max_valid_lengths = int(valid_lengths.max().item())
        if max_valid_lengths >= max_length:
            break
        
        # Forward pass - handle different interfaces
        if model_type == "surgical":
            current_qa_inputs = {
                'input_ids': padded_input_ids[:, :max_valid_lengths],
                'attention_mask': padded_attention_mask[:, :max_valid_lengths]
            }
            logits = model(image=images, qa_inputs=current_qa_inputs)
        elif model_type == "pitlora":
            logits = model(
                image=images,
                qa_inputs_ids=padded_input_ids[:, :max_valid_lengths],
                qa_att_mask=padded_attention_mask[:, :max_valid_lengths],
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        # ========================================
        
        # Next token distribution
        last_valid_logits = logits[batch_indices, valid_lengths - 1, :]
        
        # Compute metrics if requested
        if compute_metrics:
            probs = F.softmax(last_valid_logits, dim=-1)
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
        
        # Greedy choice
        next_token_ids = torch.argmax(last_valid_logits, dim=-1)
        
        if compute_metrics:
            selected_probs = probs[batch_indices, next_token_ids]
            selected_nll = -(selected_probs + 1e-12).log()
        
        # Check EOS
        is_eos = (next_token_ids == tokenizer.eos_token_id)
        
        # Store metrics (exclude EOS steps)
        if compute_metrics:
            for i in range(batch_size):
                if (not finished[i]) and (not is_eos[i]):
                    step_entropies[i].append(float(entropy[i].item()))
                    step_nlls[i].append(float(selected_nll[i].item()))
        
        # Update finished flags
        finished |= is_eos
        
        # Write tokens
        padded_input_ids[batch_indices, valid_lengths] = next_token_ids
        padded_attention_mask[batch_indices, valid_lengths] = 1
        valid_lengths += 1
        only_answer_ids = torch.cat([only_answer_ids, next_token_ids.unsqueeze(1)], dim=1)
        
        # Stop if all finished
        if bool(finished.all().item()):
            break
    
    # Decode answers
    generated_ids_cpu = only_answer_ids.detach().cpu().tolist()
    for i in range(batch_size):
        ids = generated_ids_cpu[i]
        try:
            eos_index = ids.index(tokenizer.eos_token_id)
            ids = ids[:eos_index]
        except ValueError:
            pass
        answer = tokenizer.decode(ids, skip_special_tokens=True).strip()
        answers.append(answer)
    
    # Aggregate metrics
    if compute_metrics:
        token_entropies = {
            "avg_entropy": [float(np.mean(ent)) if len(ent) else 0.0 for ent in step_entropies],
            "max_entropy": [float(np.max(ent)) if len(ent) else 0.0 for ent in step_entropies],
            "avg_prob": [float(np.mean(nll)) if len(nll) else 0.0 for nll in step_nlls],
            "max_prob": [float(np.max(nll)) if len(nll) else 0.0 for nll in step_nlls],
        }
        return answers, token_entropies
    else:
        return answers


def generate_sampled_answer(
        images, questions, model, tokenizer, max_length, device,
        temperature=1.0, top_k=50, top_p=0.9, return_per_token=True, seed=None
    ):
    """
    Generates one answer per question with specified temperature.
    
    Returns:
        answers: List[str]
        loglik_data: List[float] or List[List[float]] depending on return_per_token
        embeddings: torch.Tensor [B, H]
    """

    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    B = len(questions)
    answers = []
    per_token_logps = [[] for _ in range(B)]
    gen_token_ids = [[] for _ in range(B)]
    
    model.eval()
    with torch.no_grad():
        # Build prompts
        prompt_texts = [f"Question: {q}\nAnswer:" for q in questions]
        prompt_inputs = tokenizer(prompt_texts, return_tensors="pt",
                                  padding='longest', add_special_tokens=False)
        
        # Pre-allocate buffers
        padded_input_ids = torch.zeros((B, max_length), dtype=torch.long, device=device)
        padded_attention_mask = torch.zeros((B, max_length), device=device)
        
        orig_len = prompt_inputs['input_ids'].size(1)
        if orig_len >= max_length:
            raise ValueError("max_length must be larger than prompt length.")
        
        padded_input_ids[:, :orig_len] = prompt_inputs['input_ids'].to(device)
        padded_attention_mask[:, :orig_len] = prompt_inputs['attention_mask'].to(device)
        
        images = images.to(device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        valid_lengths = padded_attention_mask.sum(dim=1).long()
        batch_idx = torch.arange(B, device=device)
        
        # Autoregressive generation loop
        for _ in range(max_length - orig_len):
            max_valid = int(valid_lengths.max().item())
            if max_valid >= max_length or finished.all():
                break
            
            # Forward pass
            logits, hidden_states = model(
                image=images,
                qa_inputs_ids=padded_input_ids[:, :max_valid],
                qa_att_mask=padded_attention_mask[:, :max_valid],
                output_hidden_states=True
            )            
            # Get last position logits
            last_logits = logits[batch_idx, valid_lengths - 1, :]
            
            # Apply temperature scaling
            scaled = last_logits / float(max(temperature, 1e-8))
            
            # Apply top-k filtering
            if top_k and top_k > 0 and top_k < scaled.size(-1):
                kth = torch.topk(scaled, top_k, dim=-1).values[:, -1].unsqueeze(1)
                scaled = torch.where(scaled >= kth, scaled, 
                                   torch.tensor(float('-inf'), device=device))

            # Apply top-p (nucleus) filtering
            if top_p and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled = scaled.masked_fill(indices_to_remove, float('-inf'))
            
            # Compute log probabilities
            log_probs = F.log_softmax(scaled, dim=-1)
            probs = log_probs.exp()
            
            # Sample next tokens
            next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update sequences and collect data
            for i in range(B):
                if finished[i]:
                    continue
                
                tok_id = next_token_ids[i]
                lp = log_probs[i, tok_id].item()
                per_token_logps[i].append(lp)
                gen_token_ids[i].append(int(tok_id.item()))
                
                # Update sequence
                tpos = valid_lengths[i].item()
                padded_input_ids[i, tpos] = tok_id
                padded_attention_mask[i, tpos] = 1
                valid_lengths[i] += 1
                
                # Check for EOS
                if tok_id.item() == tokenizer.eos_token_id:
                    finished[i] = True
        
        # Decode answers
        for i in range(B):
            ids = gen_token_ids[i]
            if tokenizer.eos_token_id in ids:
                eos_pos = ids.index(tokenizer.eos_token_id)
                ids = ids[:eos_pos]
            answers.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
        
        # Return per-token log probs or means
        if return_per_token:
            loglik_data = per_token_logps  # List of lists
        else:
            loglik_data = [float(torch.tensor(lp).mean().item()) if len(lp) else 0.0
                          for lp in per_token_logps]
    
    return answers, loglik_data


    
def generate_sampled_answer_surgical(
        images, questions, model, tokenizer, max_length, device,
        temperature=2.5, top_k=50, top_p=0.9, return_per_token=True, seed=None
    ):
    """
    Generates sampled answers from SurgicalGPTGen model.
    
    Args:
        images: Batch of images
        questions: List of question strings
        model: SurgicalGPTGen model
        tokenizer: GPT2 tokenizer
        max_length: Maximum sequence length
        device: torch device
        temperature: Sampling temperature (higher = more random)
        top_k: Top-K filtering parameter
        top_p: Nucleus sampling parameter
        return_per_token: If True, return per-token log probs; else return means
        seed: Random seed for reproducibility
    
    Returns:
        answers: List[str]
        loglik_data: List[float] or List[List[float]] depending on return_per_token
    """
    
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    
    B = len(questions)
    answers = []
    per_token_logps = [[] for _ in range(B)]
    gen_token_ids = [[] for _ in range(B)]
    
    model.eval()
    with torch.no_grad():
        # Build prompts
        prompt_texts = [f"Question: {q}\nAnswer:" for q in questions]
        prompt_inputs = tokenizer(
            prompt_texts, 
            return_tensors="pt",
            padding='longest', 
            add_special_tokens=False
        )
        
        # Pre-allocate buffers
        padded_input_ids = torch.zeros((B, max_length), dtype=torch.long, device=device)
        padded_attention_mask = torch.zeros((B, max_length), device=device)
        
        orig_len = prompt_inputs['input_ids'].size(1)
        if orig_len >= max_length:
            raise ValueError("max_length must be larger than prompt length.")
        
        padded_input_ids[:, :orig_len] = prompt_inputs['input_ids'].to(device)
        padded_attention_mask[:, :orig_len] = prompt_inputs['attention_mask'].to(device)
        
        images = images.to(device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        valid_lengths = padded_attention_mask.sum(dim=1).long()
        batch_idx = torch.arange(B, device=device)
        
        # Autoregressive generation loop
        for _ in range(max_length - orig_len):
            max_valid = int(valid_lengths.max().item())
            if max_valid >= max_length or finished.all():
                break
            
            current_qa_inputs = {
                'input_ids': padded_input_ids[:, :max_valid],
                'attention_mask': padded_attention_mask[:, :max_valid]
            }
            
            logits = model(image=images, qa_inputs=current_qa_inputs)
            
            # Get last position logits
            last_logits = logits[batch_idx, valid_lengths - 1, :]
            
            # Apply temperature scaling
            scaled = last_logits / float(max(temperature, 1e-8))
            
            # Apply top-k filtering
            if top_k and top_k > 0 and top_k < scaled.size(-1):
                kth = torch.topk(scaled, top_k, dim=-1).values[:, -1].unsqueeze(1)
                scaled = torch.where(
                    scaled >= kth, 
                    scaled, 
                    torch.tensor(float('-inf'), device=device)
                )
            
            # Apply top-p (nucleus) filtering
            if top_p and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                scaled = scaled.masked_fill(indices_to_remove, float('-inf'))
            
            # Compute log probabilities
            log_probs = F.log_softmax(scaled, dim=-1)
            probs = log_probs.exp()
            
            # Sample next tokens
            next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update sequences and collect data
            for i in range(B):
                if finished[i]:
                    continue
                
                tok_id = next_token_ids[i]
                lp = log_probs[i, tok_id].item()
                per_token_logps[i].append(lp)
                gen_token_ids[i].append(int(tok_id.item()))
                
                # Update sequence
                tpos = valid_lengths[i].item()
                padded_input_ids[i, tpos] = tok_id
                padded_attention_mask[i, tpos] = 1
                valid_lengths[i] += 1
                
                # Check for EOS
                if tok_id.item() == tokenizer.eos_token_id:
                    finished[i] = True
        
        # Decode answers
        for i in range(B):
            ids = gen_token_ids[i]
            if tokenizer.eos_token_id in ids:
                eos_pos = ids.index(tokenizer.eos_token_id)
                ids = ids[:eos_pos]
            answers.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
        
        # Return per-token log probs or means
        if return_per_token:
            loglik_data = per_token_logps  # List of lists
        else:
            loglik_data = [
                float(torch.tensor(lp).mean().item()) if len(lp) else 0.0
                for lp in per_token_logps
            ]
    
    return answers, loglik_data


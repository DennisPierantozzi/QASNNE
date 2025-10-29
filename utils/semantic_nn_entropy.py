import numpy as np
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import List, Optional
from pathlib import Path

rouge = evaluate.load('rouge', keep_in_memory=True)

def snne(similarity_matrix, labels, variant="only_denom", temperature=1.0, epsilon=1e-8, exclude_diagonal=True, weight=None):
    # Convert inputs to tensors if they are not already
    if not isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)
    
    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int64)
        
        # Ensure the labels are a column vector for broadcasting
        labels = labels.view(-1, 1)

        # Create a mask for the labels to identify dissimilar pairs
        label_mask = labels != labels.T
        label_inf = torch.zeros_like(similarity_matrix)
        label_inf[label_mask] = float('-inf')
    else:
        # No labels: don't apply label masking (treat all as same class)
        label_inf = torch.zeros_like(similarity_matrix)

    if labels is None:
        N = similarity_matrix.size(0)
        labels = torch.zeros(N, dtype=torch.long)

    # Divide the similarity matrix by temperature
    similarity_matrix = similarity_matrix / temperature
    
    if exclude_diagonal:
        # Discard self similarity
        diag_inf = torch.diag(torch.tensor(float('-inf')).expand(labels.size(0)))
        similarity_matrix = similarity_matrix + diag_inf
    
    # Use log-sum-exp trick to stabilize the computation
    # when temperature is very low
    logsumexp_numerators = torch.logsumexp(similarity_matrix + label_inf, dim=1, keepdim=True)
    logsumexp_denominators = torch.logsumexp(similarity_matrix, dim=1, keepdim=True)
    
    # Replace -inf in numerators with log(epsilon)
    # when a class has only one sample
    inf_mask = torch.isinf(logsumexp_numerators)
    logsumexp_numerators[inf_mask] = torch.log(torch.tensor(epsilon))

    # Calculate the loss
    if variant == "full":
        loss = logsumexp_numerators - logsumexp_denominators
    elif variant == "only_num":
        loss = logsumexp_numerators
    elif variant == "only_denom":
        loss = logsumexp_denominators
    elif variant == "num_minus_denom":
        loss = 2 * torch.exp(logsumexp_numerators) - torch.exp(logsumexp_denominators) + torch.exp(torch.tensor(1./temperature)) * logsumexp_numerators.size(0)
        loss = torch.log(loss)
        
    # Weighted loss
    if weight is None:
        weight = torch.ones_like(loss)
    elif weight.size() != loss.size():
        weight = weight.view(-1, 1)
        
    loss = -(loss * weight).mean()
    
    return loss


def lexical_similarity_matrix(rouge_metric, list_strings, chunk_pairs=20000, use_stemmer=True):
    """ROUGE-L per-pair similarities; batched by pair lists."""
    n = len(list_strings)
    S = torch.eye(n, dtype=torch.float32)
    pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]

    for k in range(0, len(pairs), chunk_pairs):
        batch = pairs[k:k + chunk_pairs]
        preds = [list_strings[i] for i, j in batch]
        refs  = [list_strings[j] for i, j in batch]
        out = rouge_metric.compute(
            predictions=preds,
            references=refs,
            rouge_types=["rougeL"],
            use_stemmer=use_stemmer,
            use_aggregator=False  # <-- per-pair scores
        )
        scores = out["rougeL"]  # list of floats
        for (i, j), s in zip(batch, scores):
            S[i, j] = S[j, i] = float(s)
    return S



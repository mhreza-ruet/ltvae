"""Reconstruction accuracy + VAE loss."""
from collections.abc import Iterable
import torch
import torch.nn as nn
from rdkit import Chem
import torch.nn.functional as F

def compute_accuracy_counts(pred_logits_or_tokens, target, padding_idx=0, eos_idx=None, stop_at_eos=True):
    if pred_logits_or_tokens.dim() == 3:
        pred = pred_logits_or_tokens.argmax(dim=-1)
    elif pred_logits_or_tokens.dim() == 2:
        pred = pred_logits_or_tokens
    else:
        raise ValueError("pred_logits_or_tokens must be [B,T,V] or [B,T]")
    tgt = target[:, 1:]
    T = min(pred.size(1), tgt.size(1))
    pred, tgt = pred[:, :T], tgt[:, :T]
    mask = (tgt != padding_idx)
    if eos_idx is not None:
        mask &= (tgt != eos_idx)
    if stop_at_eos and eos_idx is not None:
        eos_hits = (tgt == eos_idx)
        has_eos = eos_hits.any(dim=1)
        first_eos = torch.argmax(eos_hits.to(torch.int32), dim=1)
        lengths = torch.where(has_eos, first_eos, torch.full_like(first_eos, T))
        ar = torch.arange(T, device=tgt.device).unsqueeze(0)
        mask &= (ar < lengths.unsqueeze(1))
    total = int(mask.sum().item())
    correct = int((pred[mask] == tgt[mask]).sum().item())
    return correct, total

def compute_accuracy(pred_logits_or_tokens, target, padding_idx=0, eos_idx=None, stop_at_eos=True):
    correct, total = compute_accuracy_counts(pred_logits_or_tokens, target, padding_idx, eos_idx, stop_at_eos)
    return (correct / total) if total > 0 else 0.0


def vae_loss(pred_logits, target, mu, logvar, padding_idx=0, eos_idx=None, kl_weight=1.0, label_smoothing: float = 0.0):
    tgt = target[:, 1:]
    T = min(pred_logits.size(1), tgt.size(1))
    pred = pred_logits[:, :T]
    tgt = tgt[:, :T]
    mask = (tgt != padding_idx)
    if eos_idx is not None:
        mask &= (tgt != eos_idx)

    if mask.any():
        logp = F.log_softmax(pred, dim=-1)                            # [B,T,V]
        gold_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)      # [B,T]
        if label_smoothing > 0.0:
            V = logp.size(-1)
            uni_lp = logp.mean(dim=-1)                                # [B,T]
            nll_tok = -(1.0 - label_smoothing) * gold_lp - label_smoothing * uni_lp
        else:
            nll_tok = -gold_lp
        recon = nll_tok[mask].mean()
    else:
        recon = pred.new_tensor(0.0)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon, kl


# ───────────────────────── chemical validity ─────────────────────────
def chemical_validity_ratio(recon_smiles: list[str]) -> float:
    valid = sum(Chem.MolFromSmiles(smi) is not None for smi in recon_smiles)
    return valid / len(recon_smiles)

# ───────────────────────── Levenshtein distance ──────────────────────
def levenshtein_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = range(len(b)+1)
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert_cost  = current[j-1] + 1
            delete_cost  = previous[j] + 1
            replace_cost = previous[j-1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]

def average_levenshtein(orig_list: list[str], recon_list: list[str]) -> float:
    dists = [levenshtein_distance(o, r) for o, r in zip(orig_list, recon_list)]
    return sum(dists) / len(dists)

# ────────────────────────── novelty metrics ───────────────────────
def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES or None if invalid."""
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _prepare_smiles(smiles_iter: Iterable[str], canonicalize: bool, drop_invalid: bool) -> list[str]:
    prepared: list[str] = []
    for smi in smiles_iter:
        if canonicalize:
            can = canonicalize_smiles(smi)
            if can is None:
                if not drop_invalid:
                    prepared.append(smi)
                continue
            prepared.append(can)
        else:
            if smi is None and drop_invalid:
                continue
            prepared.append(smi)
    return prepared


def novelty_counts(generated_smiles: Iterable[str], reference_smiles: Iterable[str], *, canonicalize: bool = True, drop_invalid: bool = True) -> tuple[int, int]:
    """Return (# novel, # valid_generated) comparing generated SMILES against a reference collection."""
    gen = _prepare_smiles(generated_smiles, canonicalize=canonicalize, drop_invalid=drop_invalid)
    if not gen:
        return 0, 0
    ref_set = set(_prepare_smiles(reference_smiles, canonicalize=canonicalize, drop_invalid=drop_invalid))
    novel = sum(1 for smi in gen if smi not in ref_set)
    return novel, len(gen)


def novelty_ratio(generated_smiles: Iterable[str], reference_smiles: Iterable[str], *, canonicalize: bool = True, drop_invalid: bool = True) -> float:
    """Fraction of generated SMILES that are not present in the reference collection."""
    novel, total = novelty_counts(generated_smiles, reference_smiles, canonicalize=canonicalize, drop_invalid=drop_invalid)
    return (novel / total) if total else 0.0


def novelty_counts_from_batches(batched_smiles: Iterable[Iterable[str]], reference_smiles: Iterable[str], *, canonicalize: bool = True, drop_invalid: bool = True) -> tuple[int, int]:
    """Convenience wrapper over novelty_counts that accepts batches of SMILES."""
    aggregated: list[str] = []
    for batch in batched_smiles:
        aggregated.extend(batch)
    return novelty_counts(aggregated, reference_smiles, canonicalize=canonicalize, drop_invalid=drop_invalid)


def novelty_ratio_from_batches(batched_smiles: Iterable[Iterable[str]], reference_smiles: Iterable[str], *, canonicalize: bool = True, drop_invalid: bool = True) -> float:
    """Novelty ratio helper when SMILES are produced in batches (e.g., via DataLoader)."""
    novel, total = novelty_counts_from_batches(batched_smiles, reference_smiles, canonicalize=canonicalize, drop_invalid=drop_invalid)
    return (novel / total) if total else 0.0

# train.py  (CUDA, Apple MPS, and CPU)

import os, time
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from device_utils import get_device
from inference import tensor_to_smiles
import data_utils   as du
import dataset      as ds
import model_bs     as mdl
import metrics      as met

# ────────────────────────── dataloader builder ─────────────────────
def build_dataloaders(train_files, val_files, seq_length, batch_size, n_train, n_val,
                      token_to_idx, device, num_workers=0):
    augmentor = du.SmilesEnumerator()

    train_smiles = du.load_smiles_list(train_files, n_samples=n_train, shuffle=True)
    val_smiles   = du.load_smiles_list(val_files,   n_samples=n_val,   shuffle=False)

    # on-the-fly: train augments, val does not
    train_ds = ds.SMILESDataset(train_smiles, seq_length=seq_length, token_to_idx=token_to_idx, augmentor=augmentor, augment_train=True)
    val_ds   = ds.SMILESDataset(val_smiles, seq_length=seq_length, token_to_idx=token_to_idx, augmentor=None, augment_train=False)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, **loader_kwargs)
    val_dl   = DataLoader(val_ds, batch_size, shuffle=False, **loader_kwargs)
    return train_dl, val_dl, train_smiles, val_smiles


def _autocast_context(use_amp):
    return torch.cuda.amp.autocast() if use_amp else nullcontext()

# ────────────────────────── train one epoch ───────────────────────
def train_one_epoch(model, loader, optimizer, scaler, pad_idx, eos_idx, kl_w, device, corruption_p, label_smoothing, clip_grad, use_amp):
    model.train()
    total_loss = 0.0
    total_corr = 0
    total_tok  = 0

    for inp, tgt in loader:
        non_blocking = device.type == "cuda"
        inp = inp.to(device, non_blocking=non_blocking)
        tgt = tgt.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(use_amp):
            logits, mu, logvar = model(inp, tgt=tgt, teacher_forcing=True, corruption_p=corruption_p)
            loss, _, _ = met.vae_loss(logits, tgt, mu, logvar, padding_idx=pad_idx, eos_idx=None, kl_weight=kl_w, label_smoothing=label_smoothing)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        corr, tok = met.compute_accuracy_counts(logits, tgt, padding_idx=pad_idx, eos_idx=eos_idx, stop_at_eos=True)
        total_corr += corr
        total_tok  += tok

    avg_loss = total_loss / len(loader)
    acc = (total_corr / total_tok) if total_tok > 0 else 0.0
    return avg_loss, acc

# ---------------------------- validation (TF) ------------------------------
@torch.no_grad()
def validate_tf(model, loader, pad_idx, eos_idx, kl_w, device, label_smoothing, use_amp):
    model.eval()
    total_loss = 0.0
    total_corr = 0
    total_tok  = 0

    for inp, tgt in loader:
        non_blocking = device.type == "cuda"
        inp = inp.to(device, non_blocking=non_blocking)
        tgt = tgt.to(device, non_blocking=non_blocking)
        with _autocast_context(use_amp):
            logits, mu, logvar = model(inp, tgt=tgt, teacher_forcing=True)
            loss, _, _ = met.vae_loss(logits, tgt, mu, logvar, padding_idx=pad_idx, eos_idx=None, kl_weight=kl_w, label_smoothing=0.0)
        total_loss += loss.item()
        corr, tok = met.compute_accuracy_counts(logits, tgt, padding_idx=pad_idx, eos_idx=eos_idx, stop_at_eos=True)
        total_corr += corr
        total_tok  += tok

    avg_loss = total_loss / len(loader)
    acc = (total_corr / total_tok) if total_tok > 0 else 0.0
    return avg_loss, acc

# ----------------------- validation (beam metrics) -------------------------
@torch.no_grad()
def validate_beam_metrics(model, loader, pad_idx, idx_to_token, device, beam_size, max_len, sos_idx, eos_idx, *, reference_smiles=None, novelty_canonicalize=True, novelty_drop_invalid=True):
    model.eval()
    # unwrap DataParallel for custom method calls
    m = model.module if isinstance(model, nn.DataParallel) else model

    orig_smiles, recon_smiles = [], []
    tot_corr = tot_tok = 0
    for inp, tgt in loader:
        non_blocking = device.type == "cuda"
        inp = inp.to(device, non_blocking=non_blocking)
        tgt = tgt.to(device, non_blocking=non_blocking)
        beams = m.beam_search(inp, beam_size=beam_size, max_len=max_len, length_penalty=0.6)
        c, t = met.compute_accuracy_counts(beams, tgt, padding_idx=pad_idx, eos_idx=eos_idx, stop_at_eos=True)
        tot_corr += c; tot_tok += t
        recon_smiles.extend(tensor_to_smiles(beams, idx_to_token, pad_idx, sos_idx=sos_idx, eos_idx=eos_idx, strip_sos_if_present=True))
        orig_smiles.extend(tensor_to_smiles(tgt,   idx_to_token, pad_idx, sos_idx=sos_idx, eos_idx=eos_idx, strip_sos_if_present=True))
    valid_ratio = met.chemical_validity_ratio(recon_smiles)
    lev = met.average_levenshtein(orig_smiles, recon_smiles)
    beam_acc = (tot_corr / tot_tok) if tot_tok > 0 else 0.0

    novelty_reference = orig_smiles if reference_smiles is None else reference_smiles
    novel_count, valid_total = met.novelty_counts(recon_smiles, novelty_reference, canonicalize=novelty_canonicalize, drop_invalid=novelty_drop_invalid)
    novelty = (novel_count / valid_total) if valid_total else np.nan
    return valid_ratio, lev, beam_acc, novelty

# ----------------------------- main training -------------------------------
def run_training(cfg, token_to_idx, idx_to_token):
    required = [
        "train_files","val_files","seq_length","n_train","n_val",
        "d_model","latent_dim","n_head","dec_layers","enc_layers","ff_dim","dropout",
        "pad_idx","sos_idx","eos_idx",
        "batch","lr","epochs","early_stop",
        "kl_anneal","kl_max",
        "save_dir","beam_every","beam_size",
        "label_smoothing","corruption_p","clip_grad"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required cfg keys: {missing}")

    # ---- device and data ----
    device = get_device(cfg.get("device"))
    print(f"Using device: {device}")
    if device.type == "cuda":
        print("Visible CUDA devices:", torch.cuda.device_count())

    train_dl, val_dl, train_smiles, val_smiles = build_dataloaders(
        cfg["train_files"], cfg["val_files"], cfg["seq_length"], cfg["batch"],
        cfg["n_train"], cfg["n_val"], token_to_idx, device,
        num_workers=cfg.get("num_workers", 0))

    novelty_reference_opt = cfg.get("novelty_reference", "train")
    if novelty_reference_opt in (None, "eval", "self"):
        novelty_reference_smiles = None
    elif novelty_reference_opt == "train":
        novelty_reference_smiles = train_smiles
    elif novelty_reference_opt in ("val", "validation"):
        novelty_reference_smiles = val_smiles
    elif novelty_reference_opt in ("train+val", "val+train", "both"):
        novelty_reference_smiles = train_smiles + val_smiles
    elif isinstance(novelty_reference_opt, (list, tuple, set)):
        novelty_reference_smiles = list(novelty_reference_opt)
    else:
        raise ValueError(f"Unsupported novelty_reference option: {novelty_reference_opt}")

    # ---- model ----
    base_model = mdl.LSTM_VAE_Trans(
        vocab_size=len(token_to_idx),
        d_model=cfg["d_model"],
        latent_dim=cfg["latent_dim"],
        pad_idx=cfg["pad_idx"],
        sos_idx=cfg["sos_idx"],
        eos_idx=cfg["eos_idx"],
        enc_layers=cfg["enc_layers"],
        dec_layers=cfg["dec_layers"],
        nhead=cfg["n_head"],
        dropout=cfg["dropout"],
        max_len=cfg["seq_length"],
        dim_feedforward=cfg["ff_dim"] if "ff_dim" in cfg else None).to(device)

    # Use all GPUs if available
    model = nn.DataParallel(base_model) if device.type == "cuda" and torch.cuda.device_count() > 1 else base_model

    # ---- optim & scaler ----
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay"))
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- optional scheduler ----
    sched = None
    if "plateau" in cfg:
        p = cfg["plateau"]
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode=p["mode"], factor=p["factor"], patience=p["patience"], verbose=p["verbose"],
                 min_lr=p["min_lr"], threshold=p["threshold"], threshold_mode=p.get("threshold_mode", "rel"), cooldown=p.get("cooldown", 0), eps=p.get("eps", 1e-8))

    novelty_canonicalize = cfg.get("novelty_canonicalize", True)
    novelty_drop_invalid = cfg.get("novelty_drop_invalid", True)

    # ---- KL schedule ----
    def kl_weight(ep):
        warm = min(ep / cfg["kl_anneal"], 1.0) * cfg["kl_max"]
        plateau = cfg.get("kl_plateau_until", 15)
        return min(warm, 0.02) if ep <= plateau else warm

    best_val_loss, no_improve = float("inf"), 0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss_tf": [], "val_acc_tf": [],
        "val_valid": [], "val_lev": [], "val_acc_beam": [], "val_novelty": []}
    os.makedirs(cfg["save_dir"], exist_ok=True)
    start = time.time()

    for ep in range(1, cfg["epochs"] + 1):
        kl_w = kl_weight(ep)

        tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, scaler, pad_idx=cfg["pad_idx"], eos_idx=cfg["eos_idx"], kl_w=kl_w, device=device,
            corruption_p=cfg["corruption_p"], label_smoothing=cfg["label_smoothing"], clip_grad=cfg["clip_grad"], use_amp=use_amp)

        va_loss, va_acc = validate_tf(model, val_dl, pad_idx=cfg["pad_idx"], eos_idx=cfg["eos_idx"], kl_w=kl_w, device=device, label_smoothing=0.0, use_amp=use_amp)

        do_beam = bool(cfg.get("beam_every")) and (ep % cfg["beam_every"] == 0)
        if do_beam:
            va_valid, va_lev, va_acc_beam, va_novelty = validate_beam_metrics(
                model, val_dl, pad_idx=cfg["pad_idx"], idx_to_token=idx_to_token,
                device=device, beam_size=cfg["beam_size"], max_len=cfg["seq_length"],
                sos_idx=cfg["sos_idx"], eos_idx=cfg["eos_idx"],
                reference_smiles=novelty_reference_smiles,
                novelty_canonicalize=novelty_canonicalize,
                novelty_drop_invalid=novelty_drop_invalid)
        else:
            va_valid = va_lev = va_acc_beam = va_novelty = np.nan

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss_tf"].append(va_loss); history["val_acc_tf"].append(va_acc)
        history["val_valid"].append(va_valid); history["val_lev"].append(va_lev)
        history["val_acc_beam"].append(va_acc_beam); history["val_novelty"].append(va_novelty)

        if do_beam:
            print(f"Epoch {ep:2d}: train {tr_loss:.4f}/{tr_acc:.3f}  "
                  f"val(tf) {va_loss:.4f}/{va_acc:.3f}  "
                  f"beam_acc {va_acc_beam:.3f}  valid {va_valid:.3f}  lev {va_lev:.2f}  "
                  f"nov {va_novelty:.3f}  KL {kl_w:.2f}")
        else:
            print(f"Epoch {ep:2d}: train {tr_loss:.4f}/{tr_acc:.3f}  "
                  f"val(tf) {va_loss:.4f}/{va_acc:.3f}  KL {kl_w:.2f}")

        if sched is not None:
            sched.step(va_loss)

        min_delta = cfg.get("early_stop_min_delta")
        if best_val_loss - va_loss > min_delta:
            best_val_loss, no_improve = va_loss, 0
            to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(to_save, os.path.join(cfg["save_dir"], "best_model.pth"))
        else:
            no_improve += 1
            if no_improve >= cfg["early_stop"]:
                print("Early stopping."); break

        if "save_every" in cfg and (ep % cfg["save_every"] == 0):
            to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(to_save, os.path.join(cfg["save_dir"], f"model_epoch_{ep}.pth"))

    print(f"Total training time: {(time.time()-start)/60:.2f} min")
    return model, history

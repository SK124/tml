#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- part1.py

# This file contains the part1 code
"""

import sys
import os

import time

import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix

import torch
import torchvision.transforms as T

from scipy.stats import norm, beta

import utils # we need this

from enum import Enum 
class FineTuneType(Enum): 
    BASIC = 1
    PGD = 2
    FGSM = 3

######### Prediction Fns #########

"""
## Basic prediction function
"""
@torch.no_grad()
def basic_predict(model, x, device="cuda"):
    x = x.to(device)
    logits = model(x)
    return logits

def fine_tune(model, train_loader, device="mps", type = FineTuneType.BASIC, num_epochs = 1):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    if type == FineTuneType.BASIC:
        augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),                    
            T.RandomCrop(32, padding=4, padding_mode='reflect'),
            T.RandomRotation(degrees=10),          
        ])

        for epoch in range(num_epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                x_aug = augment(x)
                optimizer.zero_grad()
                logits = model(x_aug)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

    elif type == FineTuneType.PGD:
        epsilon = 8/255 
        alpha = 2/255 
        num_steps = 7   

        for epoch in range(num_epochs):
            for x,y in train_loader:
                x, y = x.to(device), y.to(device)
                x_adv = x.clone().detach()
                x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-epsilon, epsilon)
                x_adv = torch.clamp(x_adv, 0, 1).detach()
            
                for step in range(num_steps):
                    x_adv.requires_grad = True
                    logits = model(x_adv)
                    loss = criterion(logits, y)
                    grad = torch.autograd.grad(loss, x_adv)[0]
                    x_adv = x_adv.detach() + alpha * grad.sign()
                    perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
                    x_adv = torch.clamp(x + perturbation, 0, 1).detach()

                optimizer.zero_grad()
                logits = model(x_adv)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            
    elif type == FineTuneType.FGSM:
        epsilon = 8/255  
        for epoch in range(num_epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                x.requires_grad = True
                logits = model(x)
                loss = criterion(logits, y)
                grad = torch.autograd.grad(loss, x)[0]

                x_adv = x + epsilon * grad.sign()
                x_adv = torch.clamp(x_adv, 0, 1).detach()

                optimizer.zero_grad()   
                logits = model(x_adv)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
    model.eval()
    return model


#### TODO: implement your defense(s) as a new prediction function
#### Make sure it is compatible with the rest of the code in this file:
####    - it needs to take model, x, device
####    - it needs to return logits
#### Note: if your predict function operates on probabilities/labels (instead of logits), that is fine provided you adjust the rest of the code.
#### Put your code here

# Bringing down model accuracy too much. Loss of utility
# Run a for loop for multiple hyperparameter settings to find best one. (0.1 - 1)
@torch.no_grad()
def output_perturbation_predict(model, x, device="cuda", scale= 0.1):
    original_logits = model(x.to(device))
    noise_dist = torch.distributions.normal.Normal(0.0, scale)
    noise = noise_dist.sample(original_logits.shape).to(device)
    noisy_logits = original_logits + noise
    return noisy_logits

# Bringing down model accuracy too much. Loss of utility
@torch.no_grad()
def output_perturbation_predict(model, x, device="cuda", laplace_scale= 10):
    original_logits = model(x.to(device))
    noise_dist = torch.distributions.laplace.Laplace(0.0, laplace_scale)
    noise = noise_dist.sample(original_logits.shape).to(device)
    noisy_logits = original_logits + noise

    return noisy_logits

# TP and FP 0, leading to NaN adv acc. Adv acc not improved much.
@torch.no_grad()
def input_perturbation_predict(model, x, device="cuda", temp=1.5, input_noise_sigma=0.01):
    x = x.to(device)
    # Add light input noise (on normalized scale)
    if input_noise_sigma > 0:
        noise = torch.randn_like(x) * input_noise_sigma
        x = x + noise
    logits = model(x)
    scaled_logits = logits / temp  # Temperature scaling
    #Add softmax temperature scaling
    scaled_logits = scaled_logits.softmax(dim=1)
    return scaled_logits

# adv acc is increased but attack acc is increased too. Takes about 15 mins to run. But performs better without finetuning.
@torch.no_grad()
def test_time_augmentation_predict(model, x, device="cuda", num_augmentations=7):
    augment = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomCrop(32, padding=6)])
    logits_sum = torch.zeros((x.size(0), 10), device=device)
    for _ in range(num_augmentations):
        aug_x = augment(x.to(device))
        logits = model(aug_x)
        logits_sum += logits
    return logits_sum / num_augmentations

#Adv acc is slightly increased but attack acc isn't reduced much
@torch.no_grad()
def gaussian_noise_predict(model, x, device="cuda", sigma=0.05):
    x = x.to(device)
    # Add Gaussian noise
    noise = torch.randn_like(x) * sigma
    noisy_x = x + noise  
    logits = model(noisy_x)
    return logits

@torch.no_grad()
def adaptive_noise_injection_defense(
    model,
    x,
    device="cuda",
    noise_strength=0.5,
    threshold=10.0,
    min_noise_std=0.0
):
    # Move input to the specified device
    x = x.to(device)

    # Get the original logits from the model
    original_logits = model(x)

    # Get the top logit for each sample in the batch
    top_logits, _ = torch.max(original_logits, dim=1, keepdim=True)

    # Calculate the noise standard deviation. The noise is scaled based on the
    # top logit value. This is the "adaptive" part of the defense.
    # We use a non-linear function like max(0, top_logit - threshold) to only
    # scale the noise for high-confidence predictions.
    noise_std = torch.clamp(top_logits - threshold, min=0.0) * noise_strength + min_noise_std

    # Generate a noise tensor from a standard normal distribution
    noise_tensor = torch.randn_like(original_logits)

    # Scale the noise tensor by the adaptive standard deviation
    scaled_noise = noise_tensor * noise_std

    # Add the scaled noise to the original logits
    noisy_logits = original_logits + scaled_noise
    return noisy_logits

######### Membership Inference Attacks (MIAs) #########

"""
## A very simple confidence threshold-based MIA
"""
@torch.no_grad()
def simple_conf_threshold_mia(predict_fn, x, thresh=0.999, device="cuda"):   
    # import pdb
    # pdb.set_trace()
    pred_y = predict_fn(x, device).cpu()
    pred_y_probas = torch.softmax(pred_y, dim=1).numpy()
    pred_y_conf = np.max(pred_y_probas, axis=-1)
    return (pred_y_conf > thresh).astype(int)
    
    
"""
## A very simple logit threshold-based MIA
"""
@torch.no_grad()
def simple_logits_threshold_mia(predict_fn, x, thresh=11, device="cuda"):   
    # import pdb
    # pdb.set_trace()
    pred_y = predict_fn(x, device).cpu().numpy()
    pred_y_max_logit = np.max(pred_y, axis=-1)
    return (pred_y_max_logit > thresh).astype(int)
    
#### TODO [optional] implement new MIA attacks.
#### Put your code here
  
######### Adversarial Examples #########
  
#### TODO [optional] implement new adversarial examples attacks.
#### Put your code here  
#### Note: you should have your code save the data to file so it can be loaded and evaluated in Main() (see below).

def load_and_grab(fp, name, num_batches=4, batch_size=256, shuffle=True):
    loader = utils.make_loader(fp, f"{name}_x", f"{name}_y", batch_size=batch_size, shuffle=shuffle)
    utils.check_loader(loader)
    
    return utils.grab_from_loader(loader, num_batches=num_batches)
    

def load_advex(fp):
    data = np.load(fp)
    return data['adv_x'], data['benign_x'], data['benign_y']
   
######### Main() #########
   
if __name__ == "__main__":

    # Let's check our software versions
    print('### Python version: ' + __import__('sys').version)
    print('### NumPy version: ' + np.__version__)
    print('### Pytorch version: ' + torch.__version__)
    print('------------')

    # deterministic seed for reproducibility
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    device = 'mps' if torch.cuda.is_available() else 'cpu'
    print(f"--- Device: {device} ---")
    print("-------------------")
    
    # keep track of time
    st = time.time()

    #### load the data
    print('\n------------ Loading Data & Model ----------')
    
    train_loader = utils.make_loader('./data/train.npz', 'train_x', 'train_y', batch_size=256, shuffle=False)
    utils.check_loader(train_loader)
    
    # val loader
    val_loader = utils.make_loader('./data/valtest.npz', 'val_x', 'val_y', batch_size=512, shuffle=False)
    utils.check_loader(val_loader)
    
    
    # create the model object   
    model_fp = './target_model.pt'
    assert os.path.exists(model_fp) # model must exist
    
    model, fg = utils.load_model(model_fp, device=device)
    assert fg == "0CCE0F932C863D6648E0", f"Modified model file {model_fp}!"
    
    st_after_model = time.time()
        
    ### let's evaluate the raw model on the train and val data
    model = fine_tune(model, train_loader, type = FineTuneType.FGSM, device=device)
    train_acc = utils.eval_model(model, train_loader, device=device)
    val_acc = utils.eval_model(model, val_loader, device=device)
    print(f"[Raw model] Train accuracy: {train_acc:.4f} ; Val accuracy: {val_acc:.4f}.")
    
    
    ### let's wrap the model prediction function so it could be replaced to implement a defense    
    ### Turn this to True to evaluate your defense (turn it back to False to see the undefended model).
    defense_enabled = True 
    if defense_enabled:
        # predict_fn = lambda x, dev: output_perturbation_predict(model, x, device=dev, scale=0.1)
        # predict_fn = lambda x, dev: output_perturbation_predict(model, x, device=dev, laplace_scale=10)
        # predict_fn = lambda x, dev: input_perturbation_predict(model, x, device=dev, temp=2.0, input_noise_sigma=0.01)
        # predict_fn = lambda x, dev: test_time_augmentation_predict(model, x, device=dev, num_augmentations=7)
        # predict_fn = lambda x, dev: gaussian_noise_predict(model, x, device=dev, sigma=0.05)
        predict_fn = lambda x, dev: adaptive_noise_injection_defense(model, x, device=dev, noise_strength=0.5, threshold=10.0, min_noise_std=0.0)
    else:
        # predict_fn points to undefended model
        predict_fn = lambda x, dev: basic_predict(model, x, device=dev)
    
    ### now let's evaluate the model with this prediction function wrapper
    train_acc = utils.eval_wrapper(predict_fn, train_loader, device=device)
    val_acc = utils.eval_wrapper(predict_fn, val_loader, device=device)
    
    print(f"[Model] Train accuracy: {train_acc:.4f} ; Val accuracy: {val_acc:.4f}.")
        
    
    ### evaluating the privacy of the model wrt membership inference
    # load the data
    in_x, in_y = load_and_grab('./data/members.npz', 'members', num_batches=2)
    out_x, out_y = load_and_grab('./data/nonmembers.npz', 'nonmembers', num_batches=2)
    
    mia_eval_x = torch.cat([in_x, out_x], 0)
    mia_eval_y = torch.cat([in_y, out_y], 0)
    mia_eval_y = mia_eval_y.cpu().detach().numpy().reshape((-1,1))
    
    assert mia_eval_x.shape[0] == mia_eval_y.shape[0]
    
    # so we can add new attack functions as needed
    print('\n------------ Privacy Attacks ----------')
    mia_attack_fns = []
    mia_attack_fns.append(('Simple Conf threshold MIA', simple_conf_threshold_mia))
    mia_attack_fns.append(('Simple Logits threshold MIA', simple_logits_threshold_mia))
    # add more lines here to add more attacks
    
    for i, tup in enumerate(mia_attack_fns):
        attack_str, attack_fn = tup
        
        in_out_preds = attack_fn(predict_fn, mia_eval_x, device=device).reshape((-1,1))       
        assert in_out_preds.shape == mia_eval_y.shape, 'Invalid attack output format'
        
        cm = confusion_matrix(mia_eval_y, in_out_preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        
        attack_acc = np.trace(cm) / np.sum(np.sum(cm))
        attack_tpr = tp / (tp + fn)
        attack_fpr = fp / (fp + tn)
        attack_adv = attack_tpr - attack_fpr
        attack_precision = tp / (tp + fp)
        attack_recall = tp / (tp + fn)
        attack_f1 = tp / (tp + 0.5*(fp + fn))
        print(f"{attack_str} --- Attack acc: {100*attack_acc:.2f}%; advantage: {attack_adv:.3f}; precision: {attack_precision:.3f}; recall: {attack_recall:.3f}; f1: {attack_f1:.3f}.")
    
    
    ### evaluating the robustness of the model wrt adversarial examples
    print('\n------------ Adversarial Examples ----------')
    advexp_fps = []
    
    advexp_fps.append(('Attack0', 'advexp0.npz', '519D7F5E79C3600B366A'))
    # uncomment/add more lines to add more attacks.
    #advexp_fps.append(('Attack1', 'advexp1.npz', None)) 
    
    for i, tup in enumerate(advexp_fps):
        attack_str, attack_fp, attack_hash = tup
        
        assert os.path.exists(attack_fp), f"Attack file {attack_fp} not found."
        _, fg = utils.memv_filehash(attack_fp)
        if attack_hash is not None:
            assert fg == attack_hash, f"Modified attack file {attack_fp}."
        
        # load the attack data
        adv_x, benign_x, benign_y = load_advex(attack_fp)
        benign_y = benign_y.flatten()
        
        benign_pred_y = predict_fn(torch.from_numpy(benign_x), device).cpu().numpy()
        benign_pred_y = np.argmax(benign_pred_y, axis=-1).astype(int)
        benign_acc = np.mean(benign_y == benign_pred_y)
        
        adv_pred_y = predict_fn(torch.from_numpy(adv_x), device).cpu().numpy()
        adv_pred_y = np.argmax(adv_pred_y, axis=-1).astype(int)
        adv_acc = np.mean(benign_y == adv_pred_y)
        
        print(f"{attack_str} [{fg}] --- Benign acc: {100*benign_acc:.2f}%; adversarial acc: {100*adv_acc:.2f}%")     
    print('------------\n')

    et = time.time()
    total_sec = et - st
    loading_sec = st_after_model - st
    
    print(f"Elapsed time -- total: {total_sec:.1f} seconds (data & model loading: {loading_sec:.1f} seconds).")
    
    sys.exit(0)
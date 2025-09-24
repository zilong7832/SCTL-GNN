# -*- coding: utf-8 -*-
import argparse
import os
from argparse import Namespace

import anndata
import mudata
import numpy as np
import pandas as pd
import torch
import anndata as ad
from scipy.sparse import issparse
from collections import Counter
from Wrapper import ScTlGNNWrapper
from Wrapper import build_pairs
from Utils import Utils
from evaluation import evaluate_prediction

DATASET = os.environ.get('DATASET', 'PBMC-Li')
REPI = int(os.environ.get("REPEAT_ID", 0))
FOLD = int(os.environ.get("FOLD_ID", 0))

def pipeline(args, inductive=False, verbose=2, logger=None, **kwargs):
    PREFIX = args.prefix
    Utils.set_seed(REPI, cuda=(args.device=="cuda"), extreme_mode=False)
    print("Preparing Data...")
    input_train, target_train, input_test, target_test = Utils.load_data(DATASET)
    
    # print(f"Subdataset creating: 5000 cells (for debugging)")
    # n_obs = input_train.n_obs
    # n_sample = min(5000, n_obs)  # 如果数据少于5000，就全取
    # sub_idx = np.random.choice(n_obs, n_sample, replace=False)
    # input_train = input_train[sub_idx, :].copy()
    # target_train = target_train[sub_idx, :].copy()
    
    # n_obs = input_test.n_obs
    # n_sample = min(5000, n_obs)  # 如果数据少于5000，就全取
    # sub_idx = np.random.choice(n_obs, n_sample, replace=False)
    # input_test = input_test[sub_idx, :].copy()
    # target_test = target_test[sub_idx, :].copy()
    
    print("Spliting data...")
    input_train, input_test, target_train, target_test = Utils.split_data(
        input_train, target_train,
        input_test=input_test,
        target_test=target_test,
        n_splits=args.n_splits,
        fold=FOLD,
        repi=REPI,
        celltype_key="celltype.l1"
    )
    
    print("Estimatimg Beta...")
    if args.ft_beta < 0:
        estimated_beta = Utils.estimate_beta_from_data(input_train, target_train, celltype_key="celltype.l1")
        print(f"Beta was not specified. Estimated beta from data: {estimated_beta:.4f}")
        args.ft_beta = estimated_beta
    else:
        print(f"Using user-specified beta: {args.ft_beta}")
    
    print("Selecting top gene...")
    input_train_filt, input_test_filt = Utils.top_correlated_genes(
        input_train, target_train, input_test, percent=15,
        celltype_key="celltype.l1", drop_zero_cells_in_train=True,
    )
    target_train = target_train[input_train_filt.obs_names, :].copy()

    print("Selecting related celltypes...")
    similarity_matrix = Utils.related_celltypes(
        input_train, celltype_key="celltype.l1", top_n_markers=50
    )

    mod1 = anndata.concat((input_train_filt, input_test_filt))
    mod2 = anndata.concat((target_train, target_test))
    mod1.var_names_make_unique(); mod2.var_names_make_unique()
    train_size = input_train_filt.n_obs
    
    g, gtest, y_train, y_test, split, data_info = Utils.prepare_graph_and_data(
        mod1, mod2, train_size, args
    )
    
    for key, value in data_info.items():
        setattr(args, key, value)
    
    test_idx = np.arange(train_size, args.CELL_SIZE)
    test_protein_names = target_test.var_names.tolist()
    
    print("\n" + "="*20 + " Pretrain " + "="*20)
    
    print("Starting pre-training...")
    model = ScTlGNNWrapper(args)
    
    if verbose > 1:
        os.makedirs(args.log_folder, exist_ok=True)
        log_filename = f"{DATASET}_{REPI}_{FOLD}.log"
        log_filepath = os.path.join(args.log_folder, log_filename)
        logger = open(log_filepath, "w")
        logger.write(str(vars(args)) + "\n")

    model.fit(g, y_train, split, not inductive, verbose, y_test, logger,
            dataset=DATASET, repi=REPI, fold=FOLD)
    
    model.save_model(DATASET, REPI, FOLD)
    
    # print("Starting Pre-train Evaluation...")
    # y_true_test = y_test
    # test_cell_names    = target_test.obs_names.tolist()
    # y_pred_pretrain = model.predict(g, test_idx, device="cpu")
    # y_true_np = y_true_test.cpu().numpy()
    # y_pred_np = y_pred_pretrain.cpu().numpy()
    
    # evaluation_results = evaluate_prediction(
    #     y_true=y_true_np,
    #     y_pred=y_pred_np,
    #     protein_names=test_protein_names,
    #     cell_names=test_cell_names
    # )
    # print("Evaluation finished:", evaluation_results)

    print("\n" + "="*20 + " Fine-tuning " + "="*20)
    
    unique_celltypes = input_train_filt.obs["celltype.l1"].unique()
    all_ft_results = {}
    celltype_counts = Counter(input_train_filt.obs["celltype.l1"].tolist())
    obs_celltypes_list = input_train_filt.obs["celltype.l1"].tolist()

    if args.lambda_anchor > 0:
        print("Building train-only anchors on CPU (KNN, cosine)...")
        with torch.no_grad():
            Z_all = model.predict(g, idx=None, device=args.device, embedding=True)  # [N,H] on GPU
        train_idx_np = np.asarray(split["train"], dtype=np.int64)
        Z_train = Z_all[train_idx_np]                                              # [N_train,H]

        # 在 CPU 上只对训练集构图，并映射回全局索引
        anchor_pairs_cpu = build_pairs(
            Z_torch=Z_train,
            k=args.anchor_k,
            tau=args.anchor_tau,
            symmetrize=True,
            index_map=train_idx_np,
        )
        model.anchor_pairs_tensor = anchor_pairs_cpu  # 先留 CPU；fine_tune 用时再搬
        del Z_all, Z_train
        torch.cuda.empty_cache()
        print(f"Anchors built: {anchor_pairs_cpu.shape[0]} edges.")

    for celltype in unique_celltypes:
        print(f"\n--- Fine-tuning for cell type: {celltype} ---")

        # Load the best pre-trained model state before each fine-tuning task
        model.load_model(DATASET, REPI, FOLD)
        
        # Step 1: Calculate sample weights using the centralized function in Utils
        # This is the single source of truth for weight calculation.
        sample_weights = Utils.loss_weights(
            obs_celltypes=obs_celltypes_list,
            target_ct=celltype,
            similarity_matrix=similarity_matrix,
            sample_counts=celltype_counts,
            beta=args.ft_beta,
            input_dim=args.hidden_size,
            device=args.device,
            gamma_threshold=args.ft_gamma_threshold,
        )
        
        # Step 2: Call the refactored fine_tune method, passing the pre-computed weights
        model.fine_tune(
            g=g,
            y=y_train,
            split=split,
            celltype=celltype,
            sample_weights=sample_weights,  # Pass the calculated weights here
            ft_epochs=args.ft_epochs,
            ft_lr=args.ft_lr,
            ft_patience=args.ft_patience,
            verbose=verbose,
            logger=logger
        )


        print(f"--- Evaluating fine-tuned model on {celltype} test cells ---")
        
        celltype_mask = (target_test.obs["celltype.l1"] == celltype).values
        
        if not np.any(celltype_mask):
            print(f"No cells of type '{celltype}' found in the test set. Skipping evaluation.")
            continue
            
        specific_test_idx = test_idx[celltype_mask]
        y_true_ft = target_test[celltype_mask, :].X
        if issparse(y_true_ft):
            y_true_ft_np = y_true_ft.toarray()
        else:
            y_true_ft_np = np.asarray(y_true_ft)
        y_pred_ft = model.predict(g, specific_test_idx, device="cpu")
        y_pred_ft_np = y_pred_ft.cpu().numpy()

        ft_eval_results = evaluate_prediction(
            y_true=y_true_ft_np,
            y_pred=y_pred_ft_np,
            protein_names=test_protein_names,
            cell_names=target_test.obs_names[celltype_mask].tolist()
        )
        print(f"Fine-tune evaluation for '{celltype}': {ft_eval_results}")
        all_ft_results[celltype] = ft_eval_results
        
        finetune_prefix = f"{DATASET}_{REPI}_{FOLD}_{celltype.replace(' ', '_')}"
        Utils.save_evaluation_results(
            results=ft_eval_results,
            output_folder=args.result_folder,
            prefix=finetune_prefix
        )

    for celltype, result in all_ft_results.items():
        print(f"Results for {celltype}: {result}")
        
    if logger is not None:
        print(f"All operations complete. Closing log file: {log_filepath}")
        logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-prefix", "--prefix", default="PBMC-Li")
    parser.add_argument("-l", "--log_folder", default="logs")
    parser.add_argument("-m", "--model_folder", default="models")
    parser.add_argument("-r", "--result_folder", default="results-weight")

    parser.add_argument("-e", "--epoch", type=int, default=15000)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5)
    parser.add_argument("-es", "--early_stopping", type=int, default=200)
    parser.add_argument("-lm", "--low_memory", type=bool, default=False)
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("-c", "--cpu", type=int, default=1)
    parser.add_argument("-lrd", "--lr_decay", type=float, default=0.99)
    parser.add_argument("-sb", "--save_best", action="store_true")
    parser.add_argument("-sf", "--save_final", action="store_true")
    parser.add_argument("-gc", "--grad_clip", type=float, default=0.0)

    parser.add_argument("-hid", "--hidden_size", type=int, default=48)
    parser.add_argument("-edd", "--edge_dropout", type=float, default=0.3)
    parser.add_argument("-mdd", "--model_dropout", type=float, default=0.2)

    parser.add_argument("-nm", "--normalization", default="group", choices=["batch", "layer", "group", "none"])
    parser.add_argument("-ac", "--activation", default="gelu", choices=["leaky_relu", "relu", "prelu", "gelu"])
    parser.add_argument("-em", "--embedding_layers", default=1, type=int, choices=[1, 2, 3])
    parser.add_argument("-ro", "--readout_layers", default=1, type=int, choices=[1, 2])
    parser.add_argument("-conv", "--conv_layers", default=4, type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("-agg", "--agg_function", default="mean", choices=["gcn", "mean"])
    parser.add_argument("-res", "--residual", default="res_cat", choices=["none", "res_add", "res_cat"])
    parser.add_argument("-inres", "--initial_residual", action="store_true")
    parser.add_argument("-nrc", "--no_readout_concatenate", action="store_true")
    parser.add_argument("-or", "--output_relu", default="none", choices=["relu", "leaky_relu", "none"])
    parser.add_argument("-ci", "--cell_init", default="none", choices=["none", "svd"])
    parser.add_argument("-ws", "--weighted_sum", action="store_true")
    parser.add_argument("-samp", "--sampling", action="store_true")
    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("-nbf", "--no_batch_features", action="store_true")

    parser.add_argument("-fte", "--ft_epochs", type=int, default=500)
    parser.add_argument("-ftlr", "--ft_lr", type=float, default=5e-4)
    parser.add_argument("-ftb", "--ft_beta", type=float, default=-1.0)
    parser.add_argument("-ftp", "--ft_patience", type=int, default=10)
    parser.add_argument("-ftgt", "--ft_gamma_threshold", type=float, default=0.2)

    parser.add_argument("--lambda_cell", type=float, default=0.3)

    parser.add_argument("--lambda_anchor", type=float, default=0.3)
    parser.add_argument("--anchor_k", type=int, default=10)
    parser.add_argument("--anchor_tau", type=float, default=0.7)

    parser.add_argument("--n_splits", type=int, default=5)

    args = parser.parse_args()
        
    base_output_dir = os.path.join("/mnt/scratch/zhan2210/output", args.prefix)
    args.log_folder = os.path.join(base_output_dir, args.log_folder)
    args.model_folder = os.path.join(base_output_dir, args.model_folder)
    args.result_folder = os.path.join(base_output_dir, args.result_folder)

    torch.set_num_threads(args.cpu)

    pipeline(args)







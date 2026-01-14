# -*- coding: utf-8 -*-
import argparse
import os

import anndata
import numpy as np
import torch
from scipy.sparse import issparse
from Wrapper import ScTlGNNWrapper
from Utils import Utils


DATASET = os.environ.get('DATASET', 'SLN111')
REPI = int(os.environ.get("REPEAT_ID", 0))
FOLD = int(os.environ.get("FOLD_ID", 0))

def pipeline(args, inductive=False, verbose=2, logger=None, **kwargs):
    Utils.set_seed(REPI, cuda=(args.device=="cuda"), extreme_mode=False)
    print("Preparing Data...")
    input_train, target_train, input_test, target_test = Utils.load_data(DATASET)
    CELLTYPE_KEY = "celltypes"

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
        celltype_key="celltypes"
    )
    
    print("Estimatimg Beta...")
    if args.ft_beta < 0:
        estimated_beta = Utils.estimate_beta_from_data(input_train, target_train, celltype_key="celltypes")
        print(f"Beta was not specified. Estimated beta from data: {estimated_beta:.4f}")
        args.ft_beta = estimated_beta
    else:
        print(f"Using user-specified beta: {args.ft_beta}")
    
    print("Selecting related celltypes...")
    similarity_matrix = Utils.related_celltypes(
        input_train, celltype_key="celltypes", top_n_markers=100, top_k=args.ft_k
    )
    
    print("Selecting features...")
    
    # 决定使用哪种模式和对应的参数值
    if args.hvg_number > 0:
        selection_mode = "hvg"
        selection_param = args.hvg_number  # 传整数，比如 2000
    else:
        selection_mode = "correlation"
        selection_param = 15               # 传百分比，比如 15 (代表15%)
    
    # 调用函数
    input_train_filt, input_test_filt = Utils.top_correlated_genes(
        input_train, target_train, input_test, 
        percent=selection_param,           # 这里传入 2000 (HVG) 或 15 (Corr)
        mode=selection_mode,               # 指定模式
        celltype_key="celltypes", 
        drop_zero_cells_in_train=True
    )
    
    # 同步 target_train (因为 top_correlated_genes 内部可能删除了全是0的细胞)
    target_train = target_train[input_train_filt.obs_names, :].copy()

    # === diagnostics: save selected genes list (v2) ===
    # input_train_filt = input_train
    # input_test_filt = input_test

    mod1 = anndata.concat((input_train_filt, input_test_filt))
    mod2 = anndata.concat((target_train, target_test))
    mod1.var_names_make_unique(); mod2.var_names_make_unique()
    train_size = input_train_filt.n_obs
    
    g, gtest, y_train, y_test, split, data_info = Utils.prepare_graph_and_data(
        mod1, mod2, train_size, args
    )
    
    # === NEW: global containers for full test-set save (Seurat-style) ===
    global_test_obs = target_test.obs.copy()
    global_var_names = target_test.var_names.tolist()

    if issparse(target_test.X):
        y_true_full = target_test.X.toarray()
    else:
        y_true_full = np.asarray(target_test.X)

    # 先用 NaN 占位，后续按 cell 名写入对应行
    y_pred_full = np.full_like(y_true_full, np.nan, dtype=np.float32)

    # 建立 “cell_name -> 全局 test 行号” 的映射
    cell_to_global = {name: i for i, name in enumerate(target_test.obs_names)}

    # === NEW: 定义 h5ad 输出目录（后面要用）===
    h5ad_out_dir = args.prediction_folder
    os.makedirs(h5ad_out_dir, exist_ok=True)
    
    for key, value in data_info.items():
        setattr(args, key, value)
    
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
        
    # === diagnostics: evaluate pretrain on global test and scatter ===
    # use graph indices consistent with mod1 = concat(train, test)
    test_idx = np.arange(train_size, data_info["CELL_SIZE"])

    try:
        y_pred_pre = model.predict(g, test_idx, device=args.device)  # g 与索引对齐更保险
        if isinstance(y_pred_pre, torch.Tensor):
            y_pred_pre_np = y_pred_pre.detach().cpu().numpy()
        else:
            y_pred_pre_np = np.asarray(y_pred_pre)
            
    except Exception as e:
        print(f"[WARN] Pretrain prediction failed: {e}")
        y_pred_pre_np = None

    if y_pred_pre_np is not None:
        if isinstance(y_test, torch.Tensor):
            y_true_pre = y_test.detach().cpu().numpy()
        elif issparse(y_test):
            y_true_pre = y_test.toarray()
        else:
            y_true_pre = np.asarray(y_test)
        
    print("\n" + "="*20 + " Fine-tuning (Corrected Strategy: Global Graph)" + "="*20)
    
    unique_celltypes = input_train_filt.obs[CELLTYPE_KEY].unique()
    
    # 注意：这里的 fine-tuning 循环已修正为使用 *全局图*
    for celltype in unique_celltypes:
        print(f"\n--- Fine-tuning for cell type: {celltype} ---")

        # 步骤 1: 获取 *全局索引* 和 *权重* (不再创建子图)
        split_sub_global, sample_weights, obs_celltypes_train_sub, celltype_counts_sub = Utils.get_finetune_indices_and_weights_new(
            input_train_full=input_train_filt,
            celltype_to_tune=celltype,
            similarity_matrix=similarity_matrix,
            repi=REPI,
            args=args,
            celltype_key=CELLTYPE_KEY
        )

        # 健壮性检查：如果没有训练索引，则跳过
        if split_sub_global is None or sample_weights is None:
            print(f"No training data for '{celltype}'. Skipping fine-tuning.")
            continue

        # 步骤 2: 重新加载预训练好的模型权重
        model.load_model(DATASET, REPI, FOLD)
        
        # 步骤 3: (已在步骤 1 中完成, sample_weights 已被计算)
        
        # # 步骤 4: 在 *全局图* 上进行微调
        # model.fine_tune(
        #     g=g,                         # <-- 使用全局图 g
        #     y=y_train,                   # <-- 使用全局标签 y_train
        #     split=split_sub_global,      # <-- 使用全局索引 split
        #     celltype=celltype,
        #     sample_weights=sample_weights, # <-- 传入已为训练集计算好的权重
        #     ft_epochs=args.ft_epochs,
        #     ft_lr=args.ft_lr,
        #     ft_patience=args.ft_patience,
        #     verbose=verbose,
        #     logger=logger
        # )
        
        # 步骤 4: 在 *全局图* 上进行微调
        model.fine_tune(
            g=g,                         # <-- 使用全局图 g
            y=y_train,                   # <-- 使用全局标签 y_train
            split=split_sub_global,      # <-- 使用全局索引 split
            celltype=celltype,
            sample_weights=sample_weights, # <-- 传入已为训练集计算好的权重
            ft_epochs=args.ft_epochs,
            ft_lr=args.ft_lr,
            ft_patience=args.ft_patience,
            verbose=verbose,
            logger=logger,
            dataset=DATASET,
            repi=REPI,
            fold=FOLD,
            k_percent=args.ft_gasdu_k_percent,
            m_period=args.ft_gasdu_M_period
        )

        print(f"--- Evaluating fine-tuned model on {celltype} test cells (in global graph) ---")
        
        # 步骤 5: 在 *全局图* 上进行评估
        global_test_indices, target_test_sub = Utils.get_finetune_test_indices_and_data(
            input_test_full=input_test_filt,
            target_test_full=target_test,
            celltype_to_tune=celltype,
            train_size=train_size,
            celltype_key=CELLTYPE_KEY
        )
        
        if global_test_indices is None:
            print(f"No test cells of type '{celltype}' found in the global test set. Skipping evaluation.")
            continue
            
        # 从 target_test_sub 中获取真实标签
        y_true_ft = target_test_sub.X
        if issparse(y_true_ft):
            y_true_ft_np = y_true_ft.toarray()
        else:
            y_true_ft_np = np.asarray(y_true_ft)
            
        # 在 *全局图 g* 上使用 *全局索引* 进行预测
        y_pred_ft = model.predict(g, global_test_indices, device=args.device)
        y_pred_ft_np = y_pred_ft.cpu().numpy()

        # 从子图的测试数据中获取正确的蛋白和细胞名称
        subgraph_cell_names = target_test_sub.obs_names.tolist()

        # ① 写回全局预测矩阵（对齐 Seurat 的整套 test）
        global_rows = np.array([cell_to_global[c] for c in subgraph_cell_names], dtype=int)
        y_pred_full[global_rows] = y_pred_ft_np
        
    # === NEW: 保存整套 test（Seurat-style）===
    gt_full_path   = os.path.join(h5ad_out_dir, f"{DATASET}_{REPI}_{FOLD}_ALL_groundtruth.h5ad")
    pred_full_path = os.path.join(h5ad_out_dir, f"{DATASET}_{REPI}_{FOLD}_ALL_prediction.h5ad")

    n_missing = int(np.isnan(y_pred_full).all(axis=1).sum())
    if n_missing > 0:
        print(f"[WARN] {n_missing} test cells not covered by any subgraph; predictions remain NaN.")

    Utils.save_matrix_as_h5ad(y_true_full, global_test_obs, global_var_names, gt_full_path)
    Utils.save_matrix_as_h5ad(y_pred_full, global_test_obs, global_var_names, pred_full_path)

    print(f"[H5AD] Full test-set saved (Seurat-style):\n  GT : {gt_full_path}\n  PRED: {pred_full_path}")

    # 关闭日志文件
    if logger is not None:
        print(f"All operations complete. Closing log file: {log_filepath}")
        logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-prefix", "--prefix", default="sctlgnn")
    parser.add_argument("-l", "--log_folder", default="logs")
    parser.add_argument("-m", "--model_folder", default="models")
    parser.add_argument("-p", "--prediction_folder", default="prediction")

    parser.add_argument("-e", "--epoch", type=int, default=15000)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2)
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
    parser.add_argument("-or", "--output_relu", default="none",
                        choices=["none", "relu", "leaky_relu", "softplus"],
                        help="Unified output activation for both training and evaluation.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Enable per-protein linear calibration (off by default).")
    parser.add_argument("-ci", "--cell_init", default="none", choices=["none", "svd"])
    parser.add_argument("-ws", "--weighted_sum", action="store_true")
    parser.add_argument("-samp", "--sampling", action="store_true")
    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("-nbf", "--no_batch_features", action="store_true")

    parser.add_argument("-fte", "--ft_epochs", type=int, default=10000)
    parser.add_argument("-ftlr", "--ft_lr", type=float, default=5e-4)
    parser.add_argument("-ftb", "--ft_beta", type=float, default=-1.0)
    parser.add_argument("-ftp", "--ft_patience", type=int, default=100)
    parser.add_argument("-ftgt", "--ft_gamma_threshold", type=float, default=0.05)
    parser.add_argument("-ftfr", "--ft_freeze_embeddings", action="store_true")
    parser.add_argument("-ftk", "--ft_k", type=int, default=2)
    
    parser.add_argument("--n_splits", type=int, default=5)
    
    parser.add_argument("--ft_gasdu_k_percent", type=float, default=0.01,
                        help="[GASDU] Percentage of params to update (e.g., 0.01). "
                             "Default: 100.0 (disables GASDU, full fine-tune).")
    parser.add_argument("--ft_gasdu_M_period", type=int, default=50,
                        help="[GASDU] Mask refresh period (in epochs). Default: 50.")
    parser.add_argument("--hvg_number", type=int, default=0, help="Number of HVGs. If 0, use correlation-based selection.")

    args = parser.parse_args()
        
    base_output_dir = os.path.join("/mnt/scratch/zhan2210/output", args.prefix)
    args.log_folder = os.path.join(base_output_dir, args.log_folder)
    args.model_folder = os.path.join(base_output_dir, args.model_folder)
    args.prediction_folder = os.path.join(base_output_dir, args.prediction_folder)

    torch.set_num_threads(args.cpu)

    pipeline(args)







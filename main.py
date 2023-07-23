import os
import argparse
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import yaml
from data_loaders import (
    MostRecentQuestionSkillDataset,
    MostEarlyQuestionSkillDataset,
)
from models.akt import AKT
from models.sakt import SAKT
from models.saint import SAINT
from train import model_train
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from utils.config import ConfigNode as CN
from utils.file_io import PathManager
from stat_data import get_stat
import wandb
import time
from time import localtime

# Directory-related exception handling
def check_directories(config):
    checkpoint_dir = config.checkpoint_dir
    model_name = config.model_name
    data_name = config.data_name

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ckpt_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, data_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

# Get dataset
def get_dataset(sequence_option):
    if sequence_option == "recent":  # the most recent N interactions
        return MostRecentQuestionSkillDataset
    elif sequence_option == "early":  # the most early N interactions
        return MostEarlyQuestionSkillDataset
    else:
        raise NotImplementedError("sequence option is not valid")

# Get model and its configuration
def get_model_info(config, train_config, device, num_questions):
    model_name = config.model_name
    data_name = config.data_name

    num_skills = train_config.num_skills
    seq_len = train_config.seq_len

    if model_name == "akt":
        model_config = config.akt_config
        model = AKT(num_skills, num_questions, seq_len, **model_config)
    elif model_name == "sakt":
        model_config = config.sakt_config
        model = SAKT(device, num_skills, num_questions,
                     seq_len, **model_config)
    elif model_name == "saint":
        model_config = config.saint_config
        model = SAINT(device, num_skills, num_questions,
                      seq_len, **model_config)
    else:
        raise NotImplementedError("model is not valid")

    return model_config, model


def get_diff_df(df, num_skills, num_questions, total_cnt_init=1, diff_unk=0.0):
    q_total_cnt = np.ones((num_questions+1))
    q_crt_cnt = np.zeros((num_questions+1))

    c_total_cnt = np.ones((num_skills+1))
    c_crt_cnt = np.zeros((num_skills+1))

    if total_cnt_init == 0:
        q_total_cnt = np.zeros((num_questions+1))
        c_total_cnt = np.zeros((num_skills+1))

    for q, c, r in zip(df["item_id"], df["skill_id"], df["correct"]):
        c_total_cnt[c] += 1
        if r:
            c_crt_cnt[c] += 1
        q_total_cnt[q] += 1
        if r:
            q_crt_cnt[q] += 1

    if diff_unk != 0.0:  # else unk is zero
        q_crt_cnt[np.where(q_total_cnt == total_cnt_init)] = diff_unk
        c_crt_cnt[np.where(c_total_cnt == total_cnt_init)] = diff_unk

    if total_cnt_init == 0:
        q_total_cnt = np.where(q_total_cnt == 0, 1, q_total_cnt)
        c_total_cnt = np.where(c_total_cnt == 0, 1, c_total_cnt)

    q_diff = q_crt_cnt/q_total_cnt
    c_diff = c_crt_cnt/c_total_cnt
    df = df.assign(item_diff=q_diff[np.array(df["item_id"].values)])
    df = df.assign(skill_diff=c_diff[np.array(df["skill_id"].values)])
    print(f"mean of skill correct ratio:{np.mean(c_diff)}")

    return df

# Get dataloader
def get_data_loader(model_name, accelerator, train_dataset, valid_dataset, test_dataset, train_config):

    batch_size = train_config.batch_size
    eval_batch_size = train_config.eval_batch_size

    train_loader = accelerator.prepare(
        DataLoader(train_dataset, batch_size=batch_size)
    )

    valid_loader = accelerator.prepare(
        DataLoader(valid_dataset, batch_size=eval_batch_size)
    )

    test_loader = accelerator.prepare(
        DataLoader(test_dataset, batch_size=eval_batch_size)
    )

def get_datasets(users, train_ids, test_ids, df, num_skills, num_questions, seq_len, dataset, config):
    np.random.shuffle(train_users)
    offset = int(len(train_ids) * 0.9)

    train_users = train_users[train_ids][:offset]
    valid_users = train_users[offset:]
    test_users = users[test_ids]

    train_df = df[df["user_id"].isin(train_users)]
    valid_df = df[df["user_id"].isin(valid_users)]
    test_df = df[df["user_id"].isin(test_users)]

    train_df = get_diff_df(train_df, num_skills, num_questions,
                            total_cnt_init=config.total_cnt_init, diff_unk=config.diff_unk)
    valid_df = get_diff_df(valid_df, num_skills, num_questions,
                            total_cnt_init=config.total_cnt_init, diff_unk=config.diff_unk)
    test_df = get_diff_df(test_df, num_skills, num_questions,
                            total_cnt_init=config.total_cnt_init, diff_unk=config.diff_unk)

    train_dataset = dataset(train_df, seq_len, num_skills, num_questions)

    valid_dataset = dataset(valid_df, seq_len, num_skills, num_questions)
    valid_dataset.sdiff_array = train_dataset.sdiff_array
    valid_dataset.qdiff_array = train_dataset.qdiff_array

    test_dataset = dataset(test_df, seq_len, num_skills, num_questions)
    test_dataset.sdiff_array = train_dataset.sdiff_array
    test_dataset.qdiff_array = train_dataset.qdiff_array

    return train_dataset, valid_dataset, test_dataset

def main(config):
    tm = localtime(time.time())
    params_str = f'{tm.tm_mon}{tm.tm_mday}{tm.tm_hour}{tm.tm_min}{tm.tm_sec}'

    if config.use_wandb:
        wandb.init(project="MKT_grad", entity="skewondr")
        wandb.run.name = params_str
        wandb.run.save()

    accelerator = Accelerator()
    device = accelerator.device

    # Load configurations
    model_name = config.model_name
    dataset_path = config.dataset_path
    data_name = config.data_name  # Change to dataset_name!
    seed = config.seed
    train_config = config.train_config

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataframe
    df_path = os.path.join(os.path.join(
        dataset_path, data_name), "preprocessed_df.csv")
    df = pd.read_csv(df_path, sep="\t")

    # 
    print("skill_min", df["skill_id"].min())
    users = df["user_id"].unique()
    np.random.shuffle(users)
    get_stat(data_name, df)

    df["skill_id"] += 1  # zero for padding
    df["item_id"] += 1  # zero for padding
    num_skills = df["skill_id"].max() + 1
    num_questions = df["item_id"].max() + 1

    # Exception handling related to directories
    check_directories(config)

    # Load training configurations
    learning_rate = train_config.learning_rate
    optimizer = train_config.optimizer
    seq_len = train_config.seq_len
    diff_order = train_config.diff_order

    # Get dataset
    dataset = get_dataset(train_config.sequence_option)

    # Initialization of metrics
    test_aucs, test_accs, test_rmses = [], [], []

    # K-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")

    for fold, (train_ids, test_ids) in enumerate(kfold.split(users)):

        # Get model information
        model_config, model = get_model_info(
            config, train_config, device, num_questions)
        if model_name == "akt" and data_name in ["statics", "assistments15"]:
            num_questions = 0
        
        # Print configurations
        print(train_config)
        print(model_config)

        # Get train, valid, test dataset
        train_dataset, valid_dataset, test_dataset = get_datasets(users, train_ids, test_ids, df, num_skills, num_questions, seq_len, dataset, config)

        train_loader, valid_loader, test_loader = get_data_loader(
            model_name, accelerator, train_dataset, valid_dataset, test_dataset, train_config)

        n_gpu = torch.cuda.device_count()

        if n_gpu > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate,
                       weight_decay=train_config.l2)

        model, opt = accelerator.prepare(model, opt)

        test_auc, test_acc, test_rmse = model_train(
            now,
            fold,
            model,
            accelerator,
            opt,
            train_loader,
            valid_loader,
            test_loader,
            config,
            n_gpu,
        )

        test_aucs.append(test_auc)
        test_accs.append(test_acc)
        test_rmses.append(test_rmse)

    test_auc = np.mean(test_aucs)
    test_acc = np.mean(test_accs)
    test_rmse = np.mean(test_rmses)

    if config.use_wandb:
        print_args = model_config.copy()
        print_args["diff_order"] = diff_order
        print_args["Model"] = model_name
        print_args["Dataset"] = data_name
        print_args["auc"] = round(test_auc, 4)
        print_args["acc"] = round(test_acc, 4)
        print_args["rmse"] = round(test_rmse, 4)
        print_args["describe"] = train_config.describe
        wandb.log(print_args)

    print("\n5-fold CV Result")
    print("AUC\tACC\tRMSE")
    print("{:.5f}\t{:.5f}\t{:.5f}".format(test_auc, test_acc, test_rmse))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="cl4kt",
        help="The name of the model to train. \
            The possible models are in [akt, cl4kt]. \
            The default model is cl4kt.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="algebra05",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--reg_cl",
        type=float,
        default=0.1,
        help="regularization parameter contrastive learning loss",
    )
    parser.add_argument(
        "--reg_l",
        type=float,
        default=0.1,
        help="regularization parameter learning loss",
    )
    parser.add_argument("--mask_prob", type=float,
                        default=0.2, help="mask probability")
    parser.add_argument("--crop_prob", type=float,
                        default=0.3, help="crop probability")
    parser.add_argument(
        "--permute_prob", type=float, default=0.3, help="permute probability"
    )
    parser.add_argument(
        "--replace_prob", type=float, default=0.3, help="replace probability"
    )
    parser.add_argument(
        "--negative_prob",
        type=float,
        default=1.0,
        help="reverse responses probability for hard negative pairs",
    )
    parser.add_argument(
        "--inter_lambda", type=float, default=1, help="loss lambda ratio for regularization"
    )
    parser.add_argument(
        "--ques_lambda", type=float, default=1, help="loss lambda ratio for regularization"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout probability"
    )
    parser.add_argument(
        "--batch_size", type=float, default=512, help="train batch size"
    )
    parser.add_argument(
        "--only_rp", type=int, default=0, help="train with only rp model"
    )
    parser.add_argument(
        "--choose_cl", type=str, default="both", help="choose between q_cl and s_cl"
    )
    parser.add_argument(
        "--describe", type=str, default="default", help="description of the training"
    )
    parser.add_argument(
        "--diff_order", type=str, default="random", help="random/des/asc/chunk"
    )
    parser.add_argument(
        "--use_wandb", type=int, default=1
    )
    parser.add_argument("--l2", type=float, default=0.0,
                        help="l2 regularization param")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="optimizer")

    parser.add_argument("--total_cnt_init", type=int,
                        default=1, help="total_cnt_init")
    parser.add_argument("--diff_unk", type=float, default=0.0, help="diff_unk")
    args = parser.parse_args()

    base_cfg_file = PathManager.open("configs/example_opt.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.use_wandb = args.use_wandb
    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.learning_rate = args.lr
    cfg.train_config.optimizer = args.optimizer
    cfg.train_config.describe = args.describe

    cfg.total_cnt_init = args.total_cnt_init
    cfg.diff_unk = args.diff_unk

    if args.model_name == "akt":
        cfg.akt_config = cfg.akt_config[cfg.data_name]

    cfg.freeze()

    main(cfg)

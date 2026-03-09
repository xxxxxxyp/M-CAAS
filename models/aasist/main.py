"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof5_train,
                        Dataset_ASVspoof5_devNeval, genSpoof_list_asv5)
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

from tqdm import tqdm

evaluation_package_dir = os.path.join(
    os.path.dirname(__file__), "evaluation-package")
if evaluation_package_dir not in sys.path:
    sys.path.append(evaluation_package_dir)
from calculate_metrics import calculate_minDCF_EER_CLLR_actDCF

warnings.filterwarnings("ignore", category=FutureWarning)


def evaluate_with_official_package(
        cm_scores_file: Union[str, os.PathLike],
        output_file: Union[str, os.PathLike]):
    """Read CM scores and evaluate them with the ASVspoof 5 official package."""
    cm_keys = []
    cm_scores = []

    with open(cm_scores_file, "r") as score_fh:
        for line in score_fh:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            cm_keys.append(parts[-2])
            cm_scores.append(float(parts[-1]))

    if not cm_scores:
        raise ValueError("No CM scores were found in {}".format(cm_scores_file))

    cm_scores = np.asarray(cm_scores, dtype=np.float64)
    cm_keys = np.asarray(cm_keys)

    min_dcf, eer, cllr, act_dcf = calculate_minDCF_EER_CLLR_actDCF(
        cm_scores, cm_keys, output_file, printout=False)
    min_dcf = float(min_dcf)
    eer = float(eer)
    cllr = float(cllr)
    act_dcf = float(act_dcf)

    with open(output_file, "w") as f_res:
        f_res.write("\nCM SYSTEM\n")
        f_res.write("\tmin DCF \t\t= {} (min DCF for countermeasure)\n".format(
            min_dcf))
        f_res.write("\tEER\t\t= {:8.9f} % (EER for countermeasure)\n".format(
            eer * 100))
        f_res.write("\tCLLR\t\t= {:8.9f} bits (CLLR for countermeasure)\n".format(
            cllr))
        f_res.write("\tactDCF\t\t= {} (actual DCF)\n".format(act_dcf))

    return min_dcf, eer * 100, cllr, act_dcf


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]

    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof5.{}".format(track)
    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof5.dev.track1.tsv")
    eval_trial_path = (
        database_path /
        "/root/lanyun-fs/M-CAAS/data/ASVspoof5.eval.track1.tsv")

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path)
        eval_minDCF, eval_eer, eval_cllr, eval_actDCF = \
            evaluate_with_official_package(
            cm_scores_file=eval_score_path,
            output_file=model_tag/"loaded_model_minDCF_EER_CLLR_actDCF.txt")
        print("DONE.\nEval eer: {:.3f}%, eval_minDCF: {:.5f}, "
              "eval_actDCF: {:.5f}, cllr: {:.5f}".format(
                  eval_eer, eval_minDCF, eval_actDCF, eval_cllr))
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 100.
    best_eval_eer = 100.
    best_dev_minDCF = 1.
    best_eval_minDCF = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", dev_trial_path)
        dev_minDCF, dev_eer, dev_cllr, dev_actDCF = \
            evaluate_with_official_package(
            cm_scores_file=metric_path/"dev_score.txt",
            output_file=metric_path /
            "dev_minDCF_EER_CLLR_actDCF_{}epo.txt".format(epoch))
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}%, dev_minDCF:{:.5f}, "
              "dev_actDCF:{:.5f}, cllr:{:.5f}".format(
                  running_loss, dev_eer, dev_minDCF, dev_actDCF, dev_cllr))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_minDCF", dev_minDCF, epoch)
        writer.add_scalar("dev_actDCF", dev_actDCF, epoch)
        writer.add_scalar("dev_cllr", dev_cllr, epoch)

        best_dev_minDCF = min(dev_minDCF, best_dev_minDCF)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, model, device,
                                        eval_score_path, eval_trial_path)
                eval_minDCF, eval_eer, eval_cllr, eval_actDCF = \
                    evaluate_with_official_package(
                    cm_scores_file=eval_score_path,
                    output_file=metric_path /
                    "minDCF_EER_CLLR_actDCF_{:03d}epo.txt".format(epoch))

                log_items = ["epoch{:03d}".format(epoch)]
                if eval_eer < best_eval_eer:
                    log_items.append("best eer {:.4f}%".format(eval_eer))
                    best_eval_eer = eval_eer
                if eval_minDCF < best_eval_minDCF:
                    log_items.append("best minDCF {:.4f}".format(eval_minDCF))
                    best_eval_minDCF = eval_minDCF
                    torch.save(model.state_dict(),
                               model_save_path / "best.pth")
                log_items.append("actDCF {:.4f}".format(eval_actDCF))
                log_items.append("CLLR {:.4f}".format(eval_cllr))
                log_text = ", ".join(log_items)
                print(log_text)
                f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_minDCF", best_dev_minDCF, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, device, eval_score_path,
                            eval_trial_path)
    eval_minDCF, eval_eer, eval_cllr, eval_actDCF = \
        evaluate_with_official_package(
            cm_scores_file=eval_score_path,
            output_file=model_tag / "minDCF_EER_CLLR_actDCF.txt")
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}%, minDCF: {:.5f}, actDCF: {:.5f}, CLLR: {:.5f}".format(
        eval_eer, eval_minDCF, eval_actDCF, eval_cllr))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_minDCF <= best_eval_minDCF:
        best_eval_minDCF = eval_minDCF
        torch.save(model.state_dict(),
                   model_save_path / "best.pth")
    print("Exp FIN. EER: {:.3f}%, minDCF: {:.5f}, actDCF: {:.5f}, CLLR: {:.5f}".format(
        best_eval_eer, best_eval_minDCF, eval_actDCF, eval_cllr))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    database_path = Path(database_path)

    # ---------------------------------------------------------
    # 修改 1：适配 ASVspoof 5 的真实路径结构
    # 注意：请确保这些路径与你本地 SSD 上的目录名一致！
    # ---------------------------------------------------------
    trn_database_path = database_path / "flac_T"  # 训练集音频目录 (示例名)
    dev_database_path = database_path / "flac_D"  # 验证集音频目录 (示例名)
    eval_database_path = database_path / "flac_E" # 评估集音频目录 (示例名)

    trn_list_path = database_path / "ASVspoof5.train.tsv"
    dev_trial_path = database_path / "ASVspoof5.dev.track1.tsv"
    eval_trial_path = database_path / "ASVspoof5.eval.track1.tsv" # 假设的评估集协议名

    d_label_trn, file_train = genSpoof_list_asv5(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof5_train(list_IDs=file_train,
                                           labels_dict=d_label_trn, # 注意这里改成了 labels_dict
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=24,
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list_asv5(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof5_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=24)

    # 如果还没发布 eval 集，可以暂时将其注释掉或者用 dev 替代
    if eval_trial_path.exists():
        file_eval = genSpoof_list_asv5(dir_meta=eval_trial_path,
                                  is_train=False,
                                  is_eval=True)
        eval_set = Dataset_ASVspoof5_devNeval(list_IDs=file_eval,
                                                 base_dir=eval_database_path)
        eval_loader = DataLoader(eval_set,
                                 batch_size=config["batch_size"],
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True,
                                 num_workers=24)
    else:
        eval_loader = dev_loader # 找不到就用 dev 占位

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            cols = trl.strip().split()
            if len(cols) < 9:
                continue
                
            # ---------------------------------------------------------
            # 修改 2：兼容 ASVspoof 5 格式并输出供 evaluation.py 计算的格式
            # ---------------------------------------------------------
            utt_id = cols[1]
            label = cols[8] # bonafide or spoof
            assert fn == utt_id
            
            # Write format: utt_id - label score (for official ASVspoof 5 evaluation package)
            fh.write("{} - {} {}\n".format(utt_id, label, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    pbar = tqdm(trn_loader, desc="Training")

    for batch_x, batch_y in pbar:

        if ii == 0:
            print("🚀 首个 Batch 已经成功从硬盘加载进了显存！")

        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        current_lr = optim.param_groups[0]['lr']
        pbar.set_postfix({"Loss": f"{batch_loss.item():.4f}", "LR": f"{current_lr:.2e}"})

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())

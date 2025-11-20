from tqdm import tqdm
import copy
import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

from utils.dataloader import load_iemocap, load_mosei, Dataloader
from utils.others import set_seed
from model import DialogueGCNModel
from BALM import GRM


def train(model: nn.Module,
          train_set: Dataloader,
          dev_set: Dataloader,
          test_set: Dataloader,
          loss_fn,
          optimizer,
          mod,
          cls_optimizer,
          logger,
          args,):

    dataset = args.dataset
    device = args.device
    best_dev_f1 = None
    best_test_f1 = None
    best_state = None
    best_epoch = None

    tau = args.tau
    early_stopping_count = 0
    current_loss = 0.0
    mod_loss = 0.0
    step = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()

        model.train()
        train_set.shuffle()

        preds, labels, losses = [], [], []
        for idx in (pbar := tqdm(range(len(train_set)), desc=f"Epoch {epoch}, CLS {current_loss:,.4f}, Mod. {mod_loss:,.4f}")):

            model.zero_grad()

            data = train_set[idx]

            length = data["length"].tolist()
            acouf = data['tensor']['a'].to(device).transpose(0, 1)
            textf = data['tensor']['t'].to(device).transpose(0, 1)
            visuf = data['tensor']['v'].to(device).transpose(0, 1)
            qmask = data['speaker_mask_tensor'].to(device).transpose(0, 1)
            label = data["label_tensor"].to(device)
            amask = data['modal_mask_tensor']['a'].to(device).transpose(0, 1)
            vmask = data['modal_mask_tensor']['v'].to(device).transpose(0, 1)
            lmask = data['modal_mask_tensor']['t'].to(device).transpose(0, 1)

            # if it is regression task log_prob=smax_feat=unprocessed output of FC, else smax_feat=softmax(logit), log_prob=log(smax_feat)
            log_prob, smax_feat, enc = model(textf, qmask, length, acouf, visuf,
                                             amask=amask, vmask=vmask, lmask=lmask)
            log_prob = log_prob.squeeze(-1)

            loss = loss_fn(log_prob, label)
            current_loss = loss.item()
            if mod is not None and mod.loss_gm is not None:
                loss += tau * mod.loss_gm
            loss.backward()

            if mod is not None:
                cls_optimizer.zero_grad()
                loss_gm, coeff_kl, cur_kl, llist = mod.get_loss(
                    enc, smax_feat, label, loss_fn, model, epoch)

                mod_loss = loss_gm.item()
                loss_a = coeff_kl[0].item()
                loss_v = coeff_kl[1].item()
                loss_l = coeff_kl[2].item()
                if logger is not None:
                    logger.log_metric('coeff_kl_a', loss_a, step=step)
                    logger.log_metric('coeff_kl_v', loss_v, step=step)
                    logger.log_metric('coeff_kl_l', loss_l, step=step)
                    logger.log_metric('kl_a', cur_kl[0].item(), step=step)
                    logger.log_metric('kl_v', cur_kl[1].item(), step=step)
                    logger.log_metric('kl_l', cur_kl[2].item(), step=step)
                    logger.log_metric('cos_a', llist[0].item(), step=step)
                    logger.log_metric('cos_v', llist[1].item(), step=step)
                    logger.log_metric('cos_l', llist[2].item(), step=step)

                cls_optimizer.step()

            step += 1
            if 'mosei' in dataset:
                preds.append(log_prob.detach().to("cpu"))
            else:
                preds.append(torch.argmax(log_prob, dim=-1).detach().to("cpu"))
            labels.append(label.to("cpu").numpy())
            losses.append(current_loss)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.grad_norm_max, norm_type=args.grad_norm)
            optimizer.step()

            pbar.set_description(
                f"Epoch {epoch}, CLS {current_loss:,.4f}, Mod. {mod_loss:,.4f}")

            del data, acouf, textf, visuf, qmask, length

        end_time = time.time()

        print(
            f"[Epoch {epoch}] [Time: {end_time - start_time}]")

        train_loss = round(np.sum(losses) / len(losses), 4)
        preds = torch.cat(preds, dim=-1).numpy()
        labels = np.concatenate(labels, axis=0)

        if 'mosei' in dataset:
            # remove neutral samples
            non_zeros = np.array([i for i, e in enumerate(labels) if e != 0])
            labels = labels[non_zeros] > 0
            preds = preds[non_zeros] > 0

        train_f1 = metrics.f1_score(labels, preds, average="weighted")
        train_acc = metrics.accuracy_score(labels, preds)
        print(
            f"[Train Loss: {train_loss}]\n[Train F1: {train_f1}]\n[Train Acc: {train_acc}]")

        dev_f1, dev_acc, dev_loss = evaluate(
            model, dev_set, loss_fn=loss_fn, logger=logger, args=args, test=False)
        print(
            f"[Dev Loss: {dev_loss}]\n[Dev F1: {dev_f1}]\n[Dev Acc: {dev_acc}]")

        if best_dev_f1 is None or dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        if args.comet:
            logger.log_metric("train_loss", train_loss, epoch=epoch)
            logger.log_metric("train_f1", train_f1, epoch=epoch)
            logger.log_metric("train_acc", train_acc, epoch=epoch)
            logger.log_metric("dev_loss", dev_loss, epoch=epoch)
            logger.log_metric("dev_f1", dev_f1, epoch=epoch)
            logger.log_metric("dev_acc", dev_acc, epoch=epoch)

        if early_stopping_count == args.early_stopping:
            print(f"Early stopping at epoch: {epoch}")
            break

    # last model
    print(f"Last model")
    f1, acc, _ = evaluate(model, test_set, loss_fn=loss_fn,
                          logger=logger, args=args, test=True)

    # best model
    print(f"Best model at epoch: {best_epoch}")
    print(f"Best dev F1: {best_dev_f1}")
    model.load_state_dict(best_state)
    f1, acc, _ = evaluate(model, test_set, loss_fn=loss_fn,
                          logger=logger, args=args, test=True)
    print(f"Best test F1: {f1}")
    print(f"Best test Acc: {acc}")

    if args.comet:
        logger.log_metric("best_test_f1", f1, epoch=epoch)
        logger.log_metric("best_test_acc", acc, epoch=epoch)

    return best_dev_f1, best_test_f1, best_state


def evaluate(model, dataset, loss_fn, args, logger=None, test=True):
    device = args.device
    model.eval()

    label_dict = args.dataset_label_dict[args.dataset]
    labels_name = list(label_dict.keys())

    with torch.no_grad():
        golds, preds, losses = [], [], []

        for idx in range(len(dataset)):
            data = dataset[idx]

            length = data["length"].tolist()
            acouf = data['tensor']['a'].to(device).transpose(0, 1)
            textf = data['tensor']['t'].to(device).transpose(0, 1)
            visuf = data['tensor']['v'].to(device).transpose(0, 1)
            qmask = data['speaker_mask_tensor'].to(device).transpose(0, 1)
            label = data["label_tensor"].to(device)
            amask = data['modal_mask_tensor']['a'].to(device).transpose(0, 1)
            vmask = data['modal_mask_tensor']['v'].to(device).transpose(0, 1)
            lmask = data['modal_mask_tensor']['t'].to(device).transpose(0, 1)

            log_prob = model(textf, qmask, length, acouf, visuf,
                             amask=amask, vmask=vmask, lmask=lmask)
            log_prob = log_prob.squeeze(-1)
            if 'mosei' in args.dataset:
                y_hat = log_prob
            else:
                y_hat = torch.argmax(log_prob, dim=-1)
            loss = loss_fn(log_prob, label) if not test else None

            golds.append(label.to("cpu"))
            preds.append(y_hat.detach().to("cpu"))
            if loss is not None:
                losses.append(loss.item())

        golds = torch.cat(golds, dim=-1).numpy()
        preds = torch.cat(preds, dim=-1).numpy()
        avg_loss = round(np.sum(losses) / len(losses), 4) if not test else None

        if 'mosei' in args.dataset:
            non_zeros = np.array(
                [i for i, e in enumerate(golds) if e != 0])  # remove 0
            golds = golds[non_zeros] > 0
            preds = preds[non_zeros] > 0

        f1 = metrics.f1_score(
            golds, preds, average="weighted", zero_division=0)
        acc = metrics.accuracy_score(golds, preds)

        if test:
            print(metrics.classification_report(
                golds, preds, target_names=labels_name, digits=4, zero_division=0))
            if logger is not None:
                logger.log_confusion_matrix(
                    golds.tolist(), preds, labels=list(labels_name), overwrite=True)

        return f1, acc, avg_loss


def get_argurment():
    parser = argparse.ArgumentParser()
    # ___________________________________ BALM Setting ______________________________________
    parser.add_argument("--tau", type=float, default=0.2,
                        help="Weight for modulation loss")
    parser.add_argument("--rho", type=float, default=1.3,
                        help="Coefficient in GRM module")
    parser.add_argument("--d_global", type=int, default=737,
                        help="Dimension of global feature in FCM module")
    parser.add_argument("--start_epoch", type=int, default=-1,
                        help="Start epoch for modulation")
    parser.add_argument("--end_epoch", type=int, default=100,
                        help="End epoch for modulation")
    parser.add_argument("--norm_modulation", action="store_true", default=False,
                        help="Whether to normalize modulation in GRM module")

    # ________________________________ Logging Setting ______________________________________
    parser.add_argument("--comet", action="store_true", default=False)
    parser.add_argument("--comet_api", type=str, default="")
    parser.add_argument("--comet_workspace", type=str, default="")
    parser.add_argument("--project_name", type=str, default="")

    # ________________________________ Trainning Setting ____________________________________
    parser.add_argument("--data_dir", type=str,
                        default="data", help="path to data folder")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoint", help="directory to save checkpoints")
    parser.add_argument("--log_path", type=str, default="log",
                        help="directory to save logs")
    parser.add_argument("--modal_MR", default=None, type=float,
                        nargs=3, help="Modal missing rate for imperfect data training in A,L,V order")
    parser.add_argument("--seed", default=12,)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["iemocap", "mosei"],
        default="iemocap",
    )
    parser.add_argument("--devset_ratio", type=float, default=0.1)
    parser.add_argument(
        "--modalities",
        type=str,
        choices=["avl", "al", "av", "lv", "a", "l", "v"],
        default="avl",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam", "adamw", "rmsprop"],
        default="adam",
    )
    parser.add_argument("--scheduler", type=str,
                        choices="reduceLR", default="reduceLR",)
    parser.add_argument("--lr", type=float, default=0.0002,)
    parser.add_argument("--early_stopping", type=int, default=-1,)
    parser.add_argument("--batch_size", type=int, default=16,)
    parser.add_argument("--epochs", type=int, default=50,)
    parser.add_argument("--device", type=str,
                        default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--grad_clipping", action="store_true", default=False,)
    parser.add_argument("--grad_norm", type=float, default=2.0,)
    parser.add_argument("--grad_norm_max", type=float, default=2.0,)

    # ________________________________ Backbone Setting ____________________________________

    parser.add_argument('--focal', type=float,
                        default=0.5, help='focal 0.5/1/2')
    parser.add_argument('--loss', default="FocalLoss",
                        help='loss function: FocalLoss/NLLLoss')
    parser.add_argument("--l2", type=float, default=0.0001,)
    parser.add_argument('--class_weight', action='store_true',
                        default=False, help='use class weights')
    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')
    parser.add_argument('--mm_fusion_mthd', default='concat_subsequently',
                        help='method to use multimodal information: mfn, concat, gated, concat_subsequently,mfn_only,tfn_only,lmf_only')
    parser.add_argument('--use_modal', action='store_true',
                        default=False, help='whether to use modal embedding')
    parser.add_argument('--use_residue', action='store_true',
                        default=True, help='whether to use residue information or not')
    parser.add_argument('--av_using_lstm', action='store_true', default=False,
                        help='whether to use lstm in acoustic and visual modality')
    parser.add_argument('--attention', default='general',
                        help='Attention type in DialogRNN model')
    parser.add_argument('--use_speaker', action='store_true',
                        default=True, help='whether to use speaker embedding')
    parser.add_argument('--dropout', type=float, default=0.4,
                        metavar='dropout', help='dropout rate')
    parser.add_argument('--alpha', type=float,
                        default=0.2, help='alpha 0.1/0.2')

    # ________________________________ End Setting ____________________________________

    args, unknown = parser.parse_known_args()
    args.embedding_dim = {
        "iemocap": {
            "a": 1582,
            "l": 1024,
            "v": 342,
        },
        "mosei": {
            "a": 512,
            "l": 1024,
            "v": 1024,
        },
    }
    args.dataset_label_dict = {
        "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
        "mosei": {
            "Negative": 0,
            "Positive": 1, },
    }
    args.n_classes = {
        "iemocap": 6,
        "mosei": 1,
    }
    args.dataset_speaker_dict = {
        "iemocap": 2,
        "mosei": 1,
    }
    args.class_weights = {
        "iemocap": torch.FloatTensor([1 / 0.086747,
                                      1 / 0.144406,
                                      1 / 0.227883,
                                      1 / 0.160585,
                                      1 / 0.127711,
                                      1 / 0.252668]),
        "mosei": torch.FloatTensor([1.0, 1.0])
    }
    if args.seed == "time":
        args.seed = int(datetime.now().timestamp())
    else:
        args.seed = int(args.seed)
    if not torch.cuda.is_available():
        args.device = "cpu"

    args.multi_modal = len(args.modalities) > 1
    return args


def main(args):
    set_seed(args.seed)
    mr = '-'.join(str(v)
                  for v in args.modal_MR) if args.modal_MR is not None else '0'
    args.model_name = f'MMGCN_{args.dataset}_{args.seed}_{args.batch_size}_{args.epochs}_{mr}'
    data_path = os.path.join(args.data_dir, f"{args.dataset}.pkl")
    class_weights = args.class_weights.get(args.dataset, None).to(args.device)
    if args.comet:
        from comet_ml import Experiment
        logger = Experiment(project_name=args.project_name,
                            api_key=args.comet_api,
                            workspace=args.comet_workspace,
                            auto_param_logging=False,
                            auto_metric_logging=False)
        logger.log_parameters(args)
    else:
        logger = None

    if args.dataset == "iemocap":
        data = load_iemocap(data_path, ratio=args.devset_ratio,
                            modal_MR=args.modal_MR)
    elif args.dataset == "mosei":
        data = load_mosei(data_path, ratio=args.devset_ratio,
                          modal_MR=args.modal_MR)

    train_set = Dataloader(data["train"], args)
    dev_set = Dataloader(data["dev"], args)
    test_set = Dataloader(data["test"], args)

    if args.modalities == "a":
        D_m = args.embedding_dim[args.dataset]["a"]
    elif args.modalities == "v":
        D_m = args.embedding_dim[args.dataset]["v"]
    else:
        D_m = args.embedding_dim[args.dataset]["l"]
    D_e = 100
    graph_h = 100
    model = DialogueGCNModel(D_m, D_e, graph_h,
                             n_speakers=args.dataset_speaker_dict[args.dataset],
                             window_past=args.windowp,
                             window_future=args.windowf,
                             n_classes=args.n_classes[args.dataset],
                             dropout=args.dropout,
                             no_cuda=args.device == "cpu",
                             alpha=args.alpha,
                             use_residue=args.use_residue,
                             D_m_v=args.embedding_dim[args.dataset]["v"],
                             D_m_a=args.embedding_dim[args.dataset]["a"],
                             modals=args.modalities,
                             att_type=args.mm_fusion_mthd,
                             av_using_lstm=args.av_using_lstm,
                             dataset=args.dataset,
                             use_speaker=args.use_speaker,
                             use_modal=args.use_modal,
                             multi_modal=args.multi_modal,
                             dim_global=args.d_global,
                             ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.l2)
    if args.dataset == 'iemocap':
        loss_fn = nn.NLLLoss(class_weights if args.class_weight else None)
    else:
        loss_fn = nn.MSELoss()

    mod, cls_optimizer = None, None
    mod = GRM(hidden_dim=300*len(args.modalities),
              output_dim=args.n_classes[args.dataset],
              modals=args.modalities,
              rho=args.rho,
              start_epoch=args.start_epoch,
              end_epoch=args.end_epoch,
              normalize=args.norm_modulation
              ).to(args.device)
    cls_optimizer = optim.Adam(mod.parameters(), lr=args.lr,
                               weight_decay=args.l2)

    dev_f1, test_f1, state = train(
        model, train_set, dev_set, test_set, loss_fn, optimizer, mod, cls_optimizer, logger, args)

    checkpoint_path = os.path.join("checkpoint", f"{args.model_name}.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({"args": args, "state_dict": state}, checkpoint_path)

    if args.comet:
        logger.log_model(name='model', file_or_folder=checkpoint_path)


if __name__ == "__main__":
    args = get_argurment()
    print(args)
    main(args)

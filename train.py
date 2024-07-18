import utils
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def train(loader, net, criterion, optimizer, epoch, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    net.train()
    for i, (input, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().to(args.device)
        target = target.float().to(args.device)

        if args.padding == 0:
            dec_inp = torch.zeros([target.shape[0], args.pred_len, target.shape[-1]]).float().to(args.device)
            dec_inp = torch.cat([target[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)
        elif args.padding == 1:
            dec_inp = torch.ones([target.shape[0], args.pred_len, target.shape[-1]]).float().to(args.device)
            dec_inp = torch.cat([target[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)
        elif args.padding == -1:
            dec_inp = target.float().to(args.device)

        #         print(args.output_attention)
        if args.output_attention:
            outputs, attens = net(input, dec_inp)
            target = target[:, -args.pred_len:, 0:].to(args.device)
        elif args.model == 'seq2seq':
            outputs = net(input, target)
        else:
            outputs = net(input, dec_inp)
        print(outputs)
        print(dec_inp.shape)
        loss = criterion(outputs, target)
        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('[train] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    logger.write([epoch, losses.avg, batch_time.avg])

    return epoch, losses.avg


def test(loader, net, criterion, epoch, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    net.eval()
    preds = []
    trues = []
    for i, (input, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().to(args.device)
        target = target.float().to(args.device)

        if args.padding == 0:
            dec_inp = torch.zeros([target.shape[0], args.pred_len, target.shape[-1]]).float().to(args.device)
            dec_inp = torch.cat([target[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)
        elif args.padding == 1:
            dec_inp = torch.ones([target.shape[0], args.pred_len, target.shape[-1]]).float().to(args.device)
            dec_inp = torch.cat([target[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)
        elif args.padding == -1:
            dec_inp = target.float().to(args.device)

        if args.output_attention:
            outputs, attens = net(input, dec_inp)
            target = target[:, -args.pred_len:, 0:].to(args.device)
        elif args.model == 'seq2seq':
            outputs = net(input, target)
        else:
            outputs = net(input, dec_inp)

        loss = criterion(outputs.detach().cpu(), target.detach().cpu())

        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('[test] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    logger.write([epoch, losses.avg, batch_time.avg])

    return epoch, losses.avg, preds, trues


def evaluate(loader, net, criterion, args):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    net.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for i, (inputs, target) in enumerate(loader):
            args.batch = inputs.shape[0]

            inputs = inputs.float().to(args.device)
            target = target.float().to(args.device)

            if args.padding == 0:
                dec_inp = torch.zeros([target.shape[0], args.pred_len, target.shape[-1]]).float().to(args.device)
            elif args.padding == 1:
                dec_inp = torch.ones([target.shape[0], args.pred_len, target.shape[-1]]).float().to(args.device)

            dec_inp = torch.cat([target[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)

            if args.output_attention:
                outputs, attens = net(inputs, dec_inp)
                target = target[:, -args.pred_len:, 0:].to(args.device)
            elif args.model == 'seq2seq':
                outputs = net(inputs, target)
            else:
                outputs = net(inputs, dec_inp)
            if i < 6:
                plt.figure(figsize=(64, 16))
                plt.plot(target.reshape(-1, 1).detach().cpu().numpy()[:1000], color='blue', alpha=0.5, linewidth=3,
                         label='input')
                plt.plot(outputs.reshape(-1, 1).detach().cpu().numpy()[:1000], color='red', alpha=0.5, linewidth=3,
                         label='output')
                plt.legend(['target', 'prediction'], prop={'size': 30})
                plt.savefig(f'{args.save_path}/pred_{i}.png')
                plt.close()

            preds = np.append(preds, outputs[:, 0, :].detach().cpu().numpy())
            trues = np.append(trues, target[:, 0, :].detach().cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print(': [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    i, len(loader), batch_time=batch_time, loss=losses))
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

    return preds, trues

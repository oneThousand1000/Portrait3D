import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--exp_dir', type=str, default='', help='for resuming train')
    parser.add_argument('--amp', dest='use_mixed_precision', action='store_true', help='use automatic mixed precision training')
    parser.add_argument('--init_scale', type=float, default=1024., help='initial loss scale')
    parser.add_argument('--cfg', type=str, default='', help='experiment configure file name')
    parser.add_argument('--data_dir', type=str, default='', help='dataset path')

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"
 
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train, exp_dir=args.exp_dir)
    cudnn.benchmark = True
    if args.cfg:
        cfg.update(args.cfg)
    if args.data_dir is not '':
        cfg.set_data_dir(args.data_dir)
    writer = SummaryWriter(cfg.log_dir)
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    scaler = amp.GradScaler(init_scale=args.init_scale, enabled=args.use_mixed_precision)
    global_step = 0
    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        print('Epoch %d/%d' % (epoch, cfg.end_epoch))
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            with amp.autocast(args.use_mixed_precision):
                loss = trainer.model(inputs, targets, meta_info, 'train')
                loss = {k: loss[k].mean() for k in loss}
                for k in loss:
                    writer.add_scalar('train/loss_' + k, loss[k].detach(), global_step)
                _loss = sum(loss[k] for k in loss)

            # backward
            with amp.autocast(False):
                _loss = scaler.scale(_loss)
                _loss.backward()
                scaler.step(trainer.optimizer)

            scaler.update(args.init_scale)

            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

            if itr % 400 == 0:
                trainer.save_visualization(inputs, targets, meta_info, epoch,itr)
            if itr % 1000 == 0:
                trainer.save_model({
                    'epoch': epoch,
                    'network': trainer.model.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                }, epoch,itr)
            global_step += 1

        # trainer.save_model({
        #     'epoch': epoch,
        #     'network': trainer.model.state_dict(),
        #     'optimizer': trainer.optimizer.state_dict(),
        # }, epoch)
        

if __name__ == "__main__":
    main()

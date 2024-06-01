import csv
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import numpy as np

from tools.data import build_dataloader
from tools.engine import Config, Trainer
from tools.utility import ArgsParser


def parse_args():
    parser = ArgsParser()
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)

    cfg.cfg['Global']['use_amp'] = False
    cfg.cfg['Global']['max_text_length'] = 200
    cfg.cfg['Architecture']['Decoder']['max_len'] = 200
    cfg.cfg['Metric']['name'] = 'RecMetricLong'
    if cfg.cfg['Global']['pretrained_model'] is None:
        cfg.cfg['Global'][
            'pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'
    trainer = Trainer(cfg, mode='eval')

    best_model_dict = trainer.status.get('metrics', {})
    trainer.logger.info('metric in ckpt ***************')
    for k, v in best_model_dict.items():
        trainer.logger.info('{}:{}'.format(k, v))

    data_dirs_list = [
        #         ['../test/IC13_857/',
        # '../test/SVT/',
        # '../test/IIIT5k/',
        # '../test/IC15_1811/',
        # '../test/SVTP/',
        # '../test/CUTE80/'],
        # [
        #     # '../test/IC13_857/',
        [
            #     './Union14M-LMDB-LongTest/general/',
            # './Union14M-LMDB-LongTest/multi_words/',
            # './Union14M-LMDB-LongTest/salient/',
            '../ultra_long/long_lmdb'
            # '../ultra_long/ultra_long_26_35_list.txt',
            # '../ultra_long/ultra_long_36_55_list.txt',
            # '../ultra_long/ultra_long_56_list.txt',
            # './long_val/ctw_1500_crop_long',
            # './long_val/u14m_train'
        ],
    ]

    cfg = cfg.cfg
    file_csv = open(
        './output/rec/' + cfg['Global']['output_dir'].split('/')[3] +
        '_result1_1_test_all_long_final_ultra_bs1.csv', 'w')
    csv_w = csv.writer(file_csv)

    for data_dirs in data_dirs_list:
        acc_each = []
        acc_each_num = []
        acc_each_dis = []
        each_long = {}
        for datadir in data_dirs:
            config_each = cfg.copy()

            config_each['Eval']['dataset']['data_dir_list'] = [datadir]
            valid_dataloader = build_dataloader(config_each, 'Eval',
                                                trainer.logger)
            trainer.logger.info(
                f'{datadir} valid dataloader has {len(valid_dataloader)} iters'
            )
            trainer.valid_dataloader = valid_dataloader
            metric = trainer.eval()
            acc_each.append(metric['acc'] * 100)
            acc_each_dis.append(metric['norm_edit_dis'])
            acc_each_num.append(metric['all_num'])

            trainer.logger.info('metric eval ***************')
            for k, v in metric.items():
                trainer.logger.info('{}:{}'.format(k, v))
                if 'each' in k:
                    csv_w.writerow([k] + v[26:])
                    each_long[k] = each_long.get(k, []) + [np.array(v[26:])]
        avg1 = np.array(acc_each) * np.array(acc_each_num) / sum(acc_each_num)
        csv_w.writerow(acc_each + [avg1.sum().tolist()] +
                       [sum(acc_each) / len(acc_each)])
        print(acc_each + [avg1.sum().tolist()] +
              [sum(acc_each) / len(acc_each)])
        avg1 = np.array(acc_each_dis) * np.array(acc_each_num) / sum(
            acc_each_num)
        csv_w.writerow(acc_each_dis + [avg1.sum().tolist()] +
                       [sum(acc_each_dis) / len(acc_each)])

        sum_all = np.array(each_long['each_len_num']).sum(0)
        for k, v in each_long.items():
            if k != 'each_len_num':
                v_sum_weight = (np.array(v) *
                                np.array(each_long['each_len_num'])).sum(0)
                sum_all_pad = np.where(sum_all == 0, 1., sum_all)
                v_all = v_sum_weight / sum_all_pad
                v_all = np.where(sum_all == 0, 0., v_all)
                csv_w.writerow([k] + v_all.tolist())
                v_26_40 = (v_all[:10] * sum_all[:10]) / sum_all[:10].sum()
                csv_w.writerow([k + '26_35'] + [v_26_40.sum().tolist()] +
                               [sum_all[:10].sum().tolist()])
                v_41_55 = (v_all[10:30] *
                           sum_all[10:30]) / sum_all[10:30].sum()
                csv_w.writerow([k + '36_55'] + [v_41_55.sum().tolist()] +
                               [sum_all[10:30].sum().tolist()])
                v_56_70 = (v_all[30:] * sum_all[30:]) / sum_all[30:].sum()
                csv_w.writerow([k + '56'] + [v_56_70.sum().tolist()] +
                               [sum_all[30:].sum().tolist()])
            else:
                csv_w.writerow([k] + sum_all.tolist())
    file_csv.close()


if __name__ == '__main__':
    main()

import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.metrics import auc
import os
from sklearn import metrics
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter

class FAR_FRR():
    def __init__(self, 
                 corr_matrix, 
                 n_sample_sequence):
        
        self.corr_matrix = corr_matrix - min(np.min(corr_matrix), 0)
        # if dist == 'L2':
        # self.corr_matrix /= 2#np.max(self.corr_matrix)

        self.thrs = [0.01 * x for x in range(100)]
        self.thr_index = 0
        self.n_sample_sequence = n_sample_sequence
        self.cat_index = 0
        self.current_cats = 0
        self.result_dict = {}
        self.roc = 0

        self._get_total_ops()

    def _get_total_ops(self):
        ops_per_class = [ns ** 2 for ns in self.n_sample_sequence]

        self.ops_total_accepted = np.sum(ops_per_class)
        self.ops_total_rejection = self.corr_matrix.shape[0] ** 2 - self.ops_total_accepted

    def _statistic_by_cat(self, dist='IP'):
        for idx, thr in enumerate(self.thrs):
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            result_per_thr = []
            for idxs in range(2, len(self.n_sample_sequence) + 1):
                start = sum(self.n_sample_sequence[:idxs - 1])
                end = sum(self.n_sample_sequence[:idxs])
                inside_band = self.corr_matrix[int(start):int(end), int(start):int(end)]
                sub_rows = self.corr_matrix[int(start):int(end), :]

                if dist == 'L2':
                    accept_inside = len(np.where(inside_band <= thr)[0])
                    accept_outside = len(np.where(sub_rows <= thr)[0]) - accept_inside

                elif dist == 'IP':
                    accept_inside = len(np.where(inside_band >= thr)[0])
                    accept_outside = len(np.where(sub_rows >= thr)[0]) - accept_inside
                else:
                    raise ValueError('dist must be L2 for euclidean distance or IP for cosine distance')

                rejected_inside = inside_band.shape[0] * inside_band.shape[1] - accept_inside
                rejected_outside = sub_rows.shape[0] * sub_rows.shape[1] - \
                                   inside_band.shape[0] * inside_band.shape[1] - accept_outside

                TP += accept_inside
                FP += accept_outside
                TN += rejected_outside
                FN += rejected_inside

            result_per_thr.append(TP)
            result_per_thr.append(FP)
            result_per_thr.append(TN)
            result_per_thr.append(FN)

            self.result_dict[str(idx)] = result_per_thr

        # self.get_far_frr()

    def get_far_frr(self, is_debug=False):
        n_thr = len(self.thrs)
        FAR = []
        FRR = []
        GAR = []
        F1 = []
        ACC = []
        for n in range(n_thr):
            res = np.array(self.result_dict[str(n)])
            TP, FP, TN, FN = res[0], res[1], res[2], res[3]
            A = TP + FP
            B = TP + FN
            C = TN + FP
            D = TN + FN
            E = A * B * C * D
            F = A + D

            # rates
            f1 = TP / (TP + 0.5 * (FP + FN))
            F1.append(f1)

            acc = (TP + TN) / F  # accuracy
            err = (FP + FN) / F  # error rate
            ACC.append(acc)

            frr = FN / B  # false rejection rate / false negative rate
            far = FP / C  # false accept rate / false positive rate
            if is_debug:
                print('thr: %.4f far: %.4f frr: %.4f acc: %.4f f1: %.4f' % (self.thrs[n], far, frr, acc, f1))
            FAR.append(far)
            FRR.append(frr)
            GAR.append(1 - frr)

        EER = [abs(FAR[i] - FRR[i]) for i in range(len(self.thrs))]
        self.eer_index = EER.index(min(EER))
        self.optimized_acc = ACC[self.eer_index]
        self.optimized_f1 = max(F1)
        self.optimized_f1_idx = F1.index(max(F1))
        self.roc = auc(FAR, GAR)
        # self.visualize(FAR, FRR, F1, ACC)
        return FAR, FRR, F1, ACC

    def visualize(self,
                  FAR,
                  FRR,
                  F1,
                  ACC,
                  save_info='',
                  is_save_fig=False):
        t = np.array(self.thrs)
        fig, ax1 = plt.subplots(dpi=200)
        color = 'tab:red'
        ax1.set_xlabel('thr')
        ax1.set_ylabel('FAR', color=color)
        ax1.plot(t, FAR, 'o-', color=color)
        ax1.plot(t, F1, 'g^-')
        ax1.plot(t, ACC, 'yv-')
        ax1.text(0.08, 0.68,
                 f'F1 {self.optimized_f1:.4f} thr: {self.thrs[self.optimized_f1_idx]: .3f} \n'
                 f'ACC {ACC[self.eer_index]: .4f} thr: {self.thrs[self.eer_index]: .3f} \n'
                 f'FAR {FAR[self.eer_index]: .4f} thr: {self.thrs[self.eer_index]: .3f} \n'
                 f'FRR {FRR[self.eer_index]: .4f} thr: {self.thrs[self.eer_index]: .3f}',
                 size=10,
                 ha="left", va="center",
                 bbox=dict(boxstyle="round",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8),
                           )
                 )

        ax1.legend(['FAR', 'F1', 'ACC'], loc='upper left')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('FRR', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, FRR, '*-', color=color)
        ax2.legend(['FRR'], loc='upper right')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.show()
        if is_save_fig:
            plt.savefig('/HDD4T/trangnt/DATA_FER/sing_data/db_llq/output/lora_llq_far_frr.png')
        else:
            plt.show()

if __name__ == '__main__':
    # init read pickle / read data
    df = pd.read_pickle('')
    df = df.sort_values(by='person_infor.name')
    # 
    features = df['feature'].values.tolist()
    features = np.array(features)
    labels = list(df['person_infor.name'].values)
    total_persons = list(np.unique(labels))
    total_counts = [labels.count(x) for x in total_persons]
    labels = Counter(labels).items()        
    
    # print(labels)
    seqs = [i[1] for i in labels]
    print(seqs)
    matrix = pairwise_distances(features, features, metric='cosine')
    print(matrix.shape)
    
    #plot
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add color bar indicating the values
    plt.title('Matrix Plot')
    plt.savefig('/HDD4T/trangnt/DATA_FER/sing_data/db_llq/output/lora_llq_matrix.png')
    
    #visualize
    farfrr = FAR_FRR(matrix, seqs)
    farfrr._statistic_by_cat(dist='L2')
    FAR, FRR, F1, ACC = farfrr.get_far_frr()
    # print(FAR, FRR, F1, ACC)
    farfrr.visualize(FAR, FRR, F1, ACC, is_save_fig=True)
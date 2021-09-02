import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, auc, roc_curve, average_precision_score

from scipy import stats
from scipy.spatial.distance import braycurtis
from skbio.stats.ordination import pcoa
from skbio.diversity.alpha import shannon as shannon_2

def shannon(abundance_list):
    return shannon_2(abundance_list, base=np.e)

def inverse_simpson_diversity(abundance_list):
    l = sum(p**2 for p in abundance_list)
    return 1.0 / l if l!=0 else 0.0

def normalize(vec):
  total = sum(vec)
  if total == 0.0:
      total = 1.0
  return [value / total for value in vec]

def normalize_dict1(d):
    def normalize(vec):
        total = sum(vec)
        if total == 0.0:
            total = 1.0
        return [value / total for value in vec]
    new_d = {}
    for k in d:
        new_d[k] = normalize(d[k])
    return new_d

def multi_boxplot(ax, abundance_dict, label_dict, pval=True, xlabels=("D", "UC"), index="Shannon Index", is_norm=False,):
    assert abundance_dict.keys() == label_dict.keys() 
    def get_alpha_diversity_df(abundance_dict, label_dict):
        sample_id_list, uc_label_list, age_label_list = [], [], []
        for k, v in label_dict.items():
            sample_id_list.append(k)
            # uc_label_list.append("UC" if v == 1 else "D")
            uc_label_list.append(xlabels[v])
            # age_label_list.append("Child" if v == 0 or v == 2 else "Adult")
        if is_norm:
            shannon_diversity = [shannon(normalize(abundance_dict[k])) for k in sample_id_list]
            inv_diversity = [inverse_simpson_diversity(normalize(abundance_dict[k])) for k in sample_id_list]
        else:
            shannon_diversity = [shannon(abundance_dict[k]) for k in sample_id_list]
            inv_diversity = [inverse_simpson_diversity(abundance_dict[k]) for k in sample_id_list]
        return pd.DataFrame(data={"sample_id": sample_id_list, "UC_label": uc_label_list, "Shannon Index": shannon_diversity, "Simpson": inv_diversity})
    
    df = get_alpha_diversity_df(abundance_dict, label_dict)
    sns.boxplot(x="UC_label", y=index, data=df, ax=ax)
    ax.set_xlabel("Phenotype", fontsize=16)
    ax.set_ylabel(index, fontsize=16)
    if pval:
        pval1 = stats.ks_2samp([row[index] for _, row in df.iterrows() if label_dict[row["sample_id"]]==0], [row[index] for _, row in df.iterrows() if label_dict[row["sample_id"]]==1]).pvalue
        # pval2 = stats.ks_2samp([row[index] for _, row in df.iterrows() if label_dict[row["sample_id"]]==2], [row[index] for _, row in df.iterrows() if label_dict[row["sample_id"]]==3]).pvalue
        # pval3 = stats.ks_2samp([row[index] for _, row in df.iterrows() if label_dict[row["sample_id"]]==0], [row[index] for _, row in df.iterrows() if label_dict[row["sample_id"]]==2]).pvalue
        # pval4 = stats.ks_2samp([row[index] for _, row in df.iterrows() if label_dict[row["sample_id"]]==1], [row[index] for _, row in df.iterrows() if label_dict[row["sample_id"]]==3]).pvalue
        # text = "\n".join(["P-Value:", "D_Child vs D_Adult: %.2e"%pval1, "UC_Child vs UC_Adult: %.2e"%pval2, "D_Child vs UC_Child: %.2e"%pval3, "D_Adult vs UC_Adult: %.2e"%pval4])
        text = "P-Value: %.2e" % pval1
        xlims = ax.get_xlim()
        xlim, ylim = ax.get_xlim()[-1], ax.get_ylim()[1]
        ax.text(xlims[0] + 0.1, ax.get_ylim()[0]+0.2, text, bbox=dict(facecolor='white', alpha=0.5))
 
    return df

def get_importance_score(abundance_dict, label_dict, n_iter=10, verbose=False, title="\n"):
    acc_list, auc_list = [], []
    x_train, y_train, sample_id_list = get_xy(abundance_dict, label_dict)
    importance_matrix = []
    healthy_score_dict = {sample_id: [] for sample_id in sample_id_list}
    for _ in range(n_iter):
        clf = RandomForestClassifier(n_estimators=500)
        clf.fit(x_train, y_train)
        y_score = clf.predict_proba(x_train)
        y_pred = clf.predict(x_train)
        importance_matrix.append(clf.feature_importances_)
        for sample_id, score in zip(sample_id_list, y_score):
            healthy_score_dict[sample_id].append(score)

    mean_score_dict = {sample_id: np.mean(np.array(score_matrix), axis=0) for sample_id, score_matrix in healthy_score_dict.items()}
    importance_matrix = np.array(importance_matrix)
    return mean_score_dict, importance_matrix

def draw_heatmap(qvalue_table, abundance_dict, label_dict, taxid_name_dict, taxid_list, use_log=True, axis_label_list = ('Healthy', 'UC'), topn = 20, n_iter=5, score_ascending=True, path=None, qval_col='q-value'):
    """
    draw_heatmap is used to display the significant of taxa.
    qvalue_table contains the qvalue of each taxa derived from differential analysis.
    """
    mean_score_dict, importance_matrix = get_importance_score(abundance_dict, label_dict, n_iter=n_iter)
    
    label_counter = Counter(label_dict.values())
    n_class = len(label_counter)
    taxid_name_score_list = []
    mean_importance_list = np.mean(importance_matrix, axis=0)
    assert len(taxid_list) == len(mean_importance_list)
    # for taxid, imp in zip(taxid_list, mean_importance_list):
#     for i, taxid in enumerate(taxid_list):
#         tax_name = taxid_name_dict.get(taxid, taxid)
#         taxid_name_score_list.append((taxid, tax_name, i, mean_importance_list[i]))
    for i, row in qvalue_table.iterrows():
        if i == topn:
            break
        taxid = row["tax_id"]
        idx = taxid_list.index(taxid)
        tax_name = taxid_name_dict.get(taxid, taxid)
        taxid_name_score_list.append((taxid, tax_name, idx, -np.log10(row[qval_col])))

    topn_taxid_name_score_list = sorted(taxid_name_score_list, key=lambda x: x[-1], reverse=True)

    width_ratios = [1] + [label_counter[i]/label_counter[0] for i in range(n_class)]
    fig, ax = plt.subplots(1, n_class+1, figsize=[8, 10], gridspec_kw={"width_ratios": width_ratios})
    plt.subplots_adjust(wspace =0.05, hspace =0)
    plt.style.use("default")
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['font.size'] = 14

    name_list = []
    imp_score_list = []
    idx_list = [item[2] for item in topn_taxid_name_score_list]
    # yerrors = np.std(known_virus_importance_matrix[:,list(reversed(selected_virus_idxs))], axis=0)
    for item in reversed(topn_taxid_name_score_list):
        name_list.append(item[1])
        imp_score_list.append(item[-1])
    # yerrors = np.std(importance_matrix[:, idx_list], axis=0)

    # cm = sns.cm.mpl_cm.BuPu_r
    # cm = sns.cm.mpl_cm.autumn
    cm = sns.cubehelix_palette(light=.7, dark=0.25, as_cmap=True)
    ax[0].errorbar(imp_score_list, range(0, topn), fmt='.', ecolor='black', color='blue')
    ax[0].grid(axis='y', linestyle='-.', color='gray')
    ax[0].set_yticks(range(0, topn))
    ax[0].set_yticklabels(name_list)
    ax[0].tick_params(labelsize=16)
    ax[0].set_xlabel("-log(%s)" % qval_col)
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(60)

    sample_order_info = [[] for _ in range(n_class)]
    abundance_matrix_list = []
    assert label_dict.keys() == mean_score_dict.keys() == abundance_dict.keys()
    for sample_id, label in label_dict.items():
        selected_abundance = [abundance_dict[sample_id][idx] for idx in idx_list]
        sample_order_info[label].append((sample_id, selected_abundance, mean_score_dict[sample_id][1]))
    for i in range(n_class):
        sample_order = sorted(sample_order_info[i], key=lambda x: x[-1], reverse=not score_ascending)
        sub_abundance_matrix = np.array([item[1] for item in sample_order])
        abundance_matrix_list.append(sub_abundance_matrix)
        cbar_kws={'label': 'Abundance'}
        if use_log:
            cbar_kws['label'] = 'log (Abundance)'
            sub_abundance_matrix = np.log2(sub_abundance_matrix+1e-20)
        
        if i == n_class - 1:
            sns.heatmap(sub_abundance_matrix.T, ax=ax[i+1], cmap=cm, cbar_kws=cbar_kws, cbar_ax=plt.axes([0.91, 0.11, 0.03, 0.77]))
        else:
            sns.heatmap(sub_abundance_matrix.T, ax=ax[i+1], cbar=False, cmap=cm)
        ax[i+1].xaxis.set_visible(False)
        ax[i+1].yaxis.set_visible(False)
        ax[i+1].set_title(axis_label_list[i])
    if path:
        plt.savefig(path, dpi=1000)
    # sns.heatmap(np.log2(first_healthy_selected_virus_abundance_matrix.T+1e-50), ax=ax[1], cbar=False, cmap=cm)
    # ax[1].xaxis.set_visible(False)
    # ax[1].yaxis.set_visible(False)
    # ax[1].set_title("Healthy")
    # sns.heatmap(np.log2(first_diseased_selected_virus_abundance_matrix.T+1e-50), ax=ax[2], cmap=cm, cbar_kws={'label': 'log (Abundance)'})
    # ax[2].xaxis.set_visible(False)
    # ax[2].yaxis.set_visible(False)
    # ax[2].set_title("Diseased")
    # plt.savefig("./figure/heatmap_dec.eps",)
    return topn_taxid_name_score_list, abundance_matrix_list

def get_xy(known_abundance_dict, label_dict, selected=None):
    assert known_abundance_dict.keys() == label_dict.keys()
    if selected is not None:
        selected_to_label = {l: i for i, l in enumerate(sorted(selected))}
    X, y = [], []
    sample_id_list = []
    if list(known_abundance_dict.keys())[0].find('-') != -1:
        id_list = sorted(known_abundance_dict.keys(), key=lambda x: int(x.split('-')[-1]))
    else:
        id_list = sorted(known_abundance_dict.keys())
    for sample_id in id_list:
        if selected is not None:
            if label_dict[sample_id] in selected:
                sample_id_list.append(sample_id)
                X.append(known_abundance_dict[sample_id])
                y.append(selected_to_label[label_dict[sample_id]])
        else:
            sample_id_list.append(sample_id)
            X.append(known_abundance_dict[sample_id])
            y.append(label_dict[sample_id])
    return np.array(X), np.array(y), sample_id_list

def get_distance_matrix(abundance_dict, label_dict, distance_func=braycurtis):
    n_sample = len(abundance_dict)
    distance_matrix = np.zeros([n_sample, n_sample])
    abundance_matrix, label_vec, sample_id_list = get_xy(abundance_dict, label_dict)
    for i in range(n_sample):
        for j in range(n_sample):
            distance = braycurtis(abundance_matrix[i, :], abundance_matrix[j, :])
            if np.isnan(distance):
                distance = 0
            distance_matrix[i, j] = distance
    return distance_matrix, abundance_matrix, label_vec, sample_id_list

def select_sample_acc_label_dict(abundance_dict, label_dict):
    new_d = {k: abundance_dict[k] for k in label_dict}
    print("before:%d\t after:%d" % (len(abundance_dict), len(new_d)))
    return new_d

def dump_shannon_index_csv(abundance_dict, label_dict, age_dict, gender_dict, path=None):
    sample_id_list = sorted(list(label_dict.keys()))
    # sample_id_list, uc_label_list, age_label_list = [], [], []
    if isinstance(next(iter(gender_dict.values())), int):
        gender_dict = {k: 'f' if v else 'm' for k, v in gender_dict.items()}
    # age_dict = {k: '<=18' if v else '>18' for k, v in age_dict.items()}
    label_dict = {k: 'UC' if v else 'Healthy' for k, v in label_dict.items()}
    shannon_diversity = [shannon(abundance_dict[k]) for k in sample_id_list]
    # inv_diversity = [inverse_simpson_diversity(abundance_dict[k]) for k in sample_id_list]
    
    t = pd.DataFrame(data={
        "SampleID": sample_id_list,
        "shannon": shannon_diversity,
        "Phenotype":[label_dict[k] for k in sample_id_list], 
        "Age": [age_dict[k] for k in sample_id_list],
        "Gender": [gender_dict[k] for k in sample_id_list]
    })
    
    if path:
        t.to_csv(path, index=False)
    return t


def dump_pcoa_csv(abundance_dict, label_dict, age_dict, gender_dict, path=None, selected_dict=None):
    if selected_dict is not None:
        abundance_dict = select_sample_acc_label_dict(abundance_dict, selected_dict)
        label_dict = select_sample_acc_label_dict(label_dict, selected_dict)
        age_dict = select_sample_acc_label_dict(age_dict, selected_dict)
        gender_dict = select_sample_acc_label_dict(gender_dict, selected_dict)
    distance_matrix, abundnace_matrix, label_vec, sample_id_list = get_distance_matrix(abundance_dict, label_dict)
    gender_dict = {k: 'f' if v else 'm' for k, v in gender_dict.items()}
    # age_dict = {k: '<=18' if v else '>18' for k, v in age_dict.items()}
    label_dict = {k: 'UC' if v else 'Healthy' for k, v in label_dict.items()}
    age_list = [age_dict[k] for k in sample_id_list]
    gender_list = [gender_dict[k] for k in sample_id_list]
    pcoa_result = pcoa(distance_matrix, number_of_dimensions=2)
    pcoa_coord = pcoa_result.samples.values
    prop_exp = pcoa_result.proportion_explained
    t = pd.DataFrame(data={
        "SampleID": sample_id_list,
        "Axis.1": pcoa_coord[:, 0],
        "Axis.2": pcoa_coord[:, 1],
        "Phenotype":[label_dict[k] for k in sample_id_list], 
        "AgeGroup": [age_dict[k] for k in sample_id_list],
        "Gender": [gender_dict[k] for k in sample_id_list]
    })
    if path:
        t.to_csv(path, index=False)
    return t, prop_exp


def pcoa_plot(ax, distance_matrix, cmap_values=None, label_vec=None, title='', legend_label_list=['Healthy_child', 'Healthy_adult', 'UC_child', 'UC_adult'], color_list = ('g', 'g', 'r', 'r'), shape_list = ('o', '^', 'o', '^'),show_cbar=False, cax=None):
    # pcoa_result = pcoa(distance_matrix, number_of_dimensions=2).samples.values
    pcoa_result = pcoa(distance_matrix, number_of_dimensions=2).samples.values
    if cmap_values is None:
        assert len(pcoa_result) == len(label_vec)
        n_class = len(set(label_vec))
        coord_division = [[] for _ in range(n_class)]
        for coord, label in zip(pcoa_result, label_vec):
            coord_division[label].append(coord)
        for i in range(n_class):
            coords = np.array(coord_division[i])
            ax.scatter(coords[:, 0], coords[:, 1], label=legend_label_list[i], c=color_list[i], marker=shape_list[i], edgecolors='black', s=80)
        ax.legend()
    else:
        cm = sns.cm.mpl_cm.autumn
        sc = ax.scatter(pcoa_result[:, 0], pcoa_result[:, 1], c=cmap_values, cmap=cm)
        if show_cbar:
            plt.colorbar(sc, cax=cax, label="Relative Abundance")
    ax.set_xlabel("PCo1")
    ax.set_ylabel('PCo2')
    ax.set_title(title, loc='left', fontdict={"fontsize": 24})
    ax.tick_params(labelsize=20)
    return pcoa_result, ax

def pcoa_plot_table(ax, t, cmap_values=None, label_vec=None, title='', legend_label_list=['Healthy', 'UC'], color_list = ('r', 'g'),show_cbar=False, cax=None):
    # pcoa_result = pcoa(distance_matrix, number_of_dimensions=2).samples.values
    # pcoa_result = pcoa(distance_matrix, number_of_dimensions=2).samples.values
    pcoa_result = t[['Axis.1', 'Axis.2']].to_numpy()
    d = {k:i for i, k in enumerate(legend_label_list)}
    label_vec = [d[row["Phenotype"]] for _, row in t.iterrows()]
    if cmap_values is None:
        assert len(pcoa_result) == len(label_vec)
        n_class = len(set(label_vec))
        coord_division = [[] for _ in range(n_class)]
        for coord, label in zip(pcoa_result, label_vec):
            coord_division[label].append(coord)
        for i in range(n_class):
            coords = np.array(coord_division[i])
            ax.scatter(coords[:, 0], coords[:, 1], label=legend_label_list[i], c=color_list[i], edgecolors='black', s=80)
        ax.legend()
    else:
        cm = sns.cm.mpl_cm.autumn
        sc = ax.scatter(pcoa_result[:, 0], pcoa_result[:, 1], c=cmap_values, cmap=cm)
        if show_cbar:
            plt.colorbar(sc, cax=cax, label="Relative Abundance")
    ax.set_xlabel("PCo1")
    ax.set_ylabel('PCo2')
    ax.set_title(title, loc='left', fontdict={"fontsize": 24})
    ax.tick_params(labelsize=20)
    return pcoa_result, ax

def multiclass_auroc_score(y_true, y_score):
    n_classes = y_score.shape[1]
    y_test = label_binarize(y_true, classes=list(range(n_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc["micro"], roc_auc["macro"]

def run_rm_cv(abundance_dict, label_dict, extra_feature_dicts=None, test_size=0.2, n_iter=10, verbose=False, selected=None, selected_feature_list=None, log_transform=False):
    """
    run_rm_cv is used to do crosse validation based on abundance dict and label dict.
    abundance_dict: {sample_id: abundance_vector}
    label_dict: {sample_id: label}
    """
    def log_trans(abundance_dict, epsilon=1e-10):
        new_dict = {}
        for k, v in abundance_dict.items():
            maxval = max(v)+epsilon
            new_dict[k] = [np.log((val+epsilon)/(maxval)) for val in v]
        return new_dict
    
    def add_extra_feature(abundance_dict, extra_feature_dict):
        new_dict = {}
        for k, v in abundance_dict.items():
            new_dict[k] = v + [extra_feature_dict[k]]
        return new_dict
    
    def normalize(vec):
        total = sum(vec)
        if total == 0.0:
            total = 1.0
        return [value / total for value in vec]
    
    if selected_feature_list:
        abundance_dict = {k: normalize([v[i] for i in selected_feature_list]) for k, v in abundance_dict.items()}
    print("Number of features: %d" % len(list(abundance_dict.values())[0]))
    n_class = len(selected) if selected is not None else len(set(label_dict.values()))
    if verbose:
        print(set(selected) if selected is not None else set(label_dict.values()))
    acc_list, roc_list = [], []
    
    if log_transform:
        abundance_dict = log_trans(abundance_dict)
    if extra_feature_dicts is not None:
        for d in extra_feature_dicts:
            abundance_dict = add_extra_feature(abundance_dict, d)
    
    pos_abundance_dict, pos_label_dict = {}, {}
    neg_abundance_dict, neg_label_dict = {}, {}
    for k, v in abundance_dict.items():
        if label_dict[k] == 1:
            pos_abundance_dict[k] = v
            pos_label_dict[k] = label_dict[k]
        else:
            neg_abundance_dict[k] = v
            neg_label_dict[k] = label_dict[k]
    assert sum(pos_label_dict.values()) == len(pos_abundance_dict)
    assert sum(neg_label_dict.values()) == 0
    print("Pos: %d, Neg: %d" % (len(pos_abundance_dict), len(neg_abundance_dict)))
    
    x_pos, y_pos, pos_sample_id_list = get_xy(pos_abundance_dict, pos_label_dict, selected)
    x_neg, y_neg, neg_sample_id_list = get_xy(neg_abundance_dict, neg_label_dict, selected)
#     importance_matrix = []
#     healthy_score_dict = {sample_id: [] for sample_id in sample_id_list}
    for i in range(n_iter):
        
        x_train_pos, x_test_pos, y_train_pos, y_test_pos = train_test_split(x_pos, y_pos, test_size=test_size)
        x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(x_neg, y_neg, test_size=test_size)
        
        x_train = np.concatenate((x_train_pos, x_train_neg), axis=0)
        x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
        y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)
        y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)
        
        x_train, y_train = shuffle(x_train, y_train)
        x_test, y_test = shuffle(x_test, y_test)
        
        if i == 0:
            print("Train pos: %d, train neg: %d" % (len(x_train_pos), len(x_train_neg)))
            print("Test pos: %d, test neg: %d" % (len(x_test_pos), len(x_test_neg)))
            print(x_train.shape, x_test.shape)
        
        clf = RandomForestClassifier(n_estimators=500)
        clf.fit(x_train, y_train)
        y_prob = clf.predict_proba(x_test)
        y_pred = clf.predict(x_test)
        # importance_matrix.append(clf.feature_importances_)
#         for sample_id, score in zip(sample_id_list, y_score):
#             healthy_score_dict[sample_id].append(score)
        acc_list.append(accuracy_score(y_test, y_pred))
        if n_class > 2:
            roc_list.append(multiclass_auroc_score(y_test, y_prob))
        else:
            roc_list.append(roc_auc_score(y_test, y_prob[:, 1]))
    print("Average accuracy: %.4f\t Standard deviation: %.4f" % (np.mean(acc_list), np.std(acc_list)))
    print("Average auroc: %.4f\t Standard deviation: %.4f" % (np.mean(roc_list), np.std(roc_list)))
#     mean_score_dict = {sample_id: np.mean(np.array(score_matrix), axis=0) for sample_id, score_matrix in healthy_score_dict.items()}
#     importance_matrix = np.array(importance_matrix)
#     return mean_score_dict, importance_matrix

def predict_score_cv(abundance_dict, label_dict, k=5, n_iter=10):
    score_dict = {k: 0 for k in label_dict.keys()}
    kf = KFold(n_splits=5, shuffle=True)
    x, y, sample_id_list = get_xy(abundance_dict, label_dict)
    for _ in range(n_iter):
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = RandomForestClassifier(n_estimators=500)
            clf.fit(x_train, y_train)
            y_prob = clf.predict_proba(x_test)
            y_pred = clf.predict(x_test)
            for i, sample_index in enumerate(test_index):
                score_dict[sample_id_list[sample_index]] += y_prob[i][1]
    return {k: v/n_iter for k, v in score_dict.items()}
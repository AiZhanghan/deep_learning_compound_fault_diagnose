'''
TODO
重构！！！
error analyze：
confusion_matrix 混淆矩阵
    plot_confusion_matrix， 可视化混淆矩阵
        plot_confusion_matrix_， 绘制
    get_confusion_df， get_confusion_df, pred_label, 涵盖所有结果
        multilabel2str
'''


import numpy as np
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt


def plot(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
        legend=None, figsize=(3.5, 2.5)):
    '''可视化

    plot(range(1, num_epochs + 1), train_acc, 'epochs', 'acc', 
         range(1, num_epochs + 1), test_acc, ['train', 'test'])
    '''
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals)
    if x2_vals:
        plt.plot(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


def confusion_matrix(y, y_hat, figsize=(12, 8)):
    '''
    混淆矩阵
    confusion_matrix： 用于绘制可视化图像，
        未在true_label中的pred_label用other表示
    confusion_df: for error analyse,
        未在true_label中的pred_label具体表示出来
    Args:
        y: pd.DataFrame, 真实标签
        y_hat: pd.DataFrame, 预测标签
        figsize: tuple(int), 显示图片大小
    '''
    # 由于K折交叉验证y_hat < label
    y = y[: len(y_hat)]
    y = y.astype(np.int)

    cm = get_confusion_matrix(y, y_hat)
    plot_confusion_matrix(cm, figsize)
    
    return cm


def get_confusion_matrix(y, y_hat):
    '''get_confusion_df, pred_label, 涵盖所有结果'''
    columns = [
        "MT", # MissingTooth
        "RC", # RootCrack
        "S", # Surface
        "CT", # ChippedTooth
        "OR", # OuterRace
        "IR", # InnerRace
        "B" # Ball
        ]
    sort_weight = {key: value for value, key in enumerate(columns, 1)}
    sort_weight['N'] = 0 # Normal
    # columns.sort()

    y_df = pd.DataFrame(y, columns=columns)
    y_df = y_df.apply(multilabel2str, axis=1)

    y_hat_df = pd.DataFrame(y_hat, columns=columns)
    y_hat_df = y_hat_df.apply(multilabel2str, axis=1)

    index = sorted(y_df.unique(), 
        key=lambda x: (len(x.split('_')), sort_weight[x.split('_')[0]], 
        0 if len(x.split("_"))== 1 else sort_weight[x.split('_')[1]]))

    columns = sorted(y_hat_df.unique(), 
        key=lambda x: (x not in index, len(x.split('_')), sort_weight[x.split('_')[0]],
        0 if len(x.split("_")) == 1 else sort_weight[x.split('_')[1]]))

    confusion_df = pd.DataFrame(np.zeros((len(index), len(columns)), 
        dtype=np.int), index=index, columns=columns)

    for i in range(len(y_df)):
        confusion_df.loc[y_df.iloc[i], y_hat_df.iloc[i]] += 1
    
    return confusion_df


def multilabel2str(x):
    '''multilabel -> str'''
    index = x.index
    res = '_'.join(list(index[x == 1]))
    if not res:
        res = 'N'
    return res
    

def plot_confusion_matrix(cm, figsize):
    '''可视化混淆矩阵'''
    set_figsize(figsize)
    
    pred_labels = list(cm.columns)
    true_labels = list(cm.index)

    cm = cm.values
    
    x_tick_marks = np.array(range(len(pred_labels))) + 0.5
    y_tick_marks = np.array(range(len(true_labels))) + 0.5
    # tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    x_ind_array = np.arange(len(pred_labels))
    y_ind_array = np.arange(len(true_labels))

    x, y = np.meshgrid(x_ind_array, y_ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c and c >= 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='green', fontsize=7, 
                va='center', ha='center')
    
    # offset the tick
    plt.gca().set_xticks(x_tick_marks, minor=True)
    plt.gca().set_yticks(y_tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    _plot_confusion_matrix(cm_normalized, pred_labels, true_labels, 
        title='Normalized confusion matrix')
    
    plt.show()


def _plot_confusion_matrix(cm, pred_labels, true_labels,
    title='Confusion Matrix', cmap=plt.cm.get_cmap('Greys')):
    '''for func plot_confusion_matrix, 外部不该调用该函数'''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(pred_labels)))
    ylocations = np.array(range(len(true_labels)))
    plt.xticks(xlocations, pred_labels, rotation=90)
    plt.yticks(ylocations, true_labels)
    plt.ylim(-0.5, 19.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def set_figsize(figsize=(3.5, 2.5)):
    '''设置图的尺寸'''
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')
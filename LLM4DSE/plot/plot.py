import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import matplotlib as mpl
import pickle

dse_KERNELS = ['doitgen', 'bicg', 'heat-3d', 'jacobi-2d']
prompt_engineer = ['few-shot', 'CoT', 'OPRO', 'best']
llm_model = ['GPT3.5_turbo', 'GPT4.1']
if __name__ == '__main__':
    dict_i = {}
    dict_j = {}
    for llm in llm_model:
        for pr in prompt_engineer:
            dict2 = {}
            for kernel in dse_KERNELS:
                with open(f'/home/wslcccc/passion/LLM4DSE/plot/ref/{kernel}.pickle', 'rb') as f:
                    data_1 = pickle.load(f)
                val_1 = []
                for i in data_1.keys():
                    val, _ = i
                    val_1.append(val)
                with open(f'/home/wslcccc/passion/LLM4DSE/plot/{llm}/{pr}/{kernel}.pickle', 'rb') as f:
                    data = pickle.load(f)
                tmp = []
                for j in data.values():
                    for k in range(len(j), 3):
                        j.append(0)
                    temp = 0
                    for num, val in enumerate(val_1):
                        temp += abs(val - j[num]) / val
                    tmp.append(temp / 3)
                tmp.sort(reverse=True)
                dict2[kernel] = tmp
            if llm == 'GPT3.5_turbo':
                dict_i[pr] = dict2
            else:
                dict_j[pr] = dict2

    # 配置科研图表风格
    plt.style.use('seaborn-v0_8-poster')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4
    })


    def plot_research_style(data_dict):
        # 获取算法和目标列表
        pr = list(data_dict.keys())
        targets = list(data_dict[pr[0]].keys())

        # 创建子图布局
        n_targets = len(targets)
        n_cols = min(2, n_targets)
        n_rows = int(np.ceil(n_targets / n_cols))

        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        axs = np.array(axs).flatten()

        # 使用科学家友好色系
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
        colors = {alg: colors[i] for i, alg in enumerate(pr)}

        # 绘制每个子图
        for idx, target in enumerate(targets):
            ax = axs[idx]
            for alg in pr:
                values = data_dict[alg][target]
                x = np.arange(len(values))
                ax.plot(x, values,
                        color=colors[alg],
                        linewidth=1.5,
                        label=alg)

            # 子图装饰
            ax.set_title(f'{target}', pad=10)
            ax.set_xlabel('The number of explored designs', labelpad=8)
            ax.set_ylabel('Mean ADRS', labelpad=8)
            ax.tick_params(axis='both', which='major', pad=6)

            # 科学计数法格式化
            if any(np.max(data_dict[alg][target]) > 1e4 for alg in pr):
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

            # 添加次要网格
            ax.grid(True, which='minor', linestyle=':', alpha=0.4)
            ax.set_axisbelow(True)

        # 隐藏多余子图
        for j in range(n_targets, len(axs)):
            axs[j].axis('off')

        # 统一图例
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc='upper center',
                   ncol=len(pr),
                   frameon=True,
                   framealpha=1.0,
                   edgecolor='w',
                   title='The convergence curves',
                   title_fontsize=20,
                   bbox_to_anchor=(0.5, 1.02 if n_rows == 1 else 1.05))

        plt.tight_layout(pad=3.0)
        return fig, axs
    fig, _ = plot_research_style(dict_i)
    plt.savefig('111.png', bbox_inches='tight', dpi=300)
    fig1, _ = plot_research_style(dict_j)
    plt.savefig('222.png', bbox_inches='tight', dpi=300)
    plt.show()
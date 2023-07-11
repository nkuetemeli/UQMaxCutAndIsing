import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.demo_max_cut import *
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

font_size = 14

params = {'axes.labelsize': font_size,
          'axes.titlesize': font_size,}
plt.rcParams.update(params)

baseline_index = 5
baseline_color = mcolors.CSS4_COLORS['cyan']

costs_color = mcolors.CSS4_COLORS['gold']
acosts_color = mcolors.CSS4_COLORS['darkolivegreen']
error_costs_color = mcolors.CSS4_COLORS['darkolivegreen']

qaoa_gradient_color = mcolors.CSS4_COLORS['gray']
qaoa_black_box_color = mcolors.CSS4_COLORS['gold']
dwave_color = mcolors.CSS4_COLORS['tomato']
mc_color = mcolors.CSS4_COLORS['darkolivegreen']

zorder_fg = 10
zorder_bg = 1

barWidth = 0.1

linewidth = 3



def vis_arccos_approx(save=False):
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = np.arccos(np.sin(x))
    y_approx = np.pi/2 - x
    plt.plot(x, y, color='y', label=rf'$f(x) = \arccos(\sin(x))$', linewidth=linewidth)
    plt.plot(x, y_approx, color=acosts_color, label=rf'$f(x) = \pi/2 - x$', linewidth=linewidth)
    plt.xlabel(rf'$x$')
    plt.ylabel(rf'$f(x)$')
    plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], [rf'$-2\pi$', rf'$-\pi$', rf'$0$', rf'$\pi$', rf'$2\pi$'])
    plt.legend(fontsize=font_size)

    if save:
        save_fig(plt.gcf(), path=path.join(results_folder, 'images/arccos_sin.pdf'))
    plt.show()
    return

def arcos_sin():
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = np.arccos(np.sin(x))
    y_approx = np.pi/2 - x
    plt.plot(x, y, label=rf'$f(x) = \arccos(\sin(x))$')
    plt.plot(x, y_approx, color='y', label=rf'$f(x) = \pi/2 - x$')
    plt.xlabel(rf'$x$')
    plt.ylabel(rf'$f(x)$')
    plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], [rf'$-2\pi$', rf'$-\pi$', rf'$0$', rf'$\pi$', rf'$2\pi$'])
    plt.legend()
    plt.show()
    return

def vis_eval_mc(save=True):
    def vis_experiment_coef_K():
        """
        Visualizes the results of experiment_coef_K
        Results are saved in the form of a dictionary {n:
                                                        {coef_K:
                                                            {costs: [[simulation 1], ..., [simulation n]],
                                                            acosts: [[simulation 1], ..., [simulation n]],
                                                            qcosts: [[simulation 1], ..., [simulation n]]}}
        """

        error_costs = {'positive_weights': None, 'positive_negative_weights': None}
        experiments = ['positive_weights', 'positive_negative_weights']
        titles = [r'$C_{ij} \in \mathbb{R}_{> 0}$', r'$C_{ij} \in \mathbb{R}$']

        fig = plt.figure()
        fig.set_size_inches(9, 5)
        fig.tight_layout(pad=3.0)

        for num_ax, experiment, title in zip(range(len(experiments)), experiments, titles):
            file_name = 'experiment_coef_K_' + experiment + '.json'
            file_name = path.join(results_folder, file_name)

            with open(file_name) as file:
                results = json.load(file)

            n = 10
            coef_Ks = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]

            costs = np.zeros((len(coef_Ks), 2 ** n))
            acosts = np.zeros((len(coef_Ks), 2 ** n))
            qcosts = np.zeros((len(coef_Ks), 2 ** n))

            for i, coef_K in enumerate(coef_Ks):
                costs[i] = np.mean(results[str(n)][str(coef_K)]['costs'], axis=0)
                acosts[i] = np.mean(results[str(n)][str(coef_K)]['acosts'], axis=0)
                qcosts[i] = np.mean(results[str(n)][str(coef_K)]['qcosts'], axis=0)

            min_costs = np.min(costs, axis=1, keepdims=True)
            max_costs = np.max(costs, axis=1, keepdims=True)

            normalize = lambda c: (c - min_costs) / (max_costs - min_costs)
            costs = normalize(costs)
            acosts = normalize(acosts)
            qcosts = normalize(qcosts)
            error_costs[experiment] = np.linalg.norm(acosts - qcosts, axis=1)

            xx, yy = np.meshgrid(np.arange(2 ** n), coef_Ks)
            xx, yy = np.flip(xx, 1), np.flip(yy, 1)

            acosts_arranged = np.array(list(map(lambda x, y: y[x], np.argsort(costs), acosts)))

            ax = fig.add_subplot(120 + num_ax + 1, projection='3d', computed_zorder=False)

            ax.plot_surface(xx, yy, np.sort(costs), color=costs_color, rstride=10, cstride=5, alpha=1,
                            zorder=zorder_bg)
            ax.plot_surface(xx, yy, acosts_arranged, color=acosts_color, rstride=10, cstride=5, alpha=.5,
                            zorder=zorder_fg)
            ax.plot(xx[baseline_index, :], yy[baseline_index, :], acosts_arranged[baseline_index, :],
                    color=baseline_color, zorder=zorder_fg, linewidth=1.5)

            ax.xaxis.labelpad = -1
            ax.yaxis.labelpad = -1
            ax.zaxis.labelpad = -1
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel(r'$Cut$')
            ax.set_ylabel(r'$\lambda$')
            ax.set_zlabel(r'$Costs$')
            ax.set_xlim(-1, 2 ** n + 1)
            ax.set_ylim(-.1, 1.1)
            ax.set_zlim(-.1, 1.1)
            ax.set_title(title)

        costs_patch = mpatches.Patch(color=costs_color, label=r'$C$: True costs')
        acosts_patch = mpatches.Patch(color=acosts_color, label=r'$K \cdot \sin(\hat C)$: Transformed costs')
        baseline_patch = mlines.Line2D([], [], color=baseline_color, label=r'Selected profile')
        fig.legend(handles=[costs_patch, acosts_patch, baseline_patch], loc='upper center', ncol=3, fontsize=font_size)

        if save:
            save_fig(fig, path=path.join('../', 'images/experiment_K.pdf'))
        plt.show()
        return

    def vis_experiment_num_shots():
        """
        Visualizes the results of experiment_num_shots
        Results are saved in the form of a dictionary {n:
                                                        {shots:
                                                            {costs: [[simulation 1], ..., [simulation n]],
                                                            acosts: [[simulation 1], ..., [simulation n]],
                                                            qcosts: [[simulation 1], ..., [simulation n]]}}
        """

        error_costs = {'positive_weights': None, 'positive_negative_weights': None}
        experiments = ['positive_weights', 'positive_negative_weights']
        titles = [r'$C_{ij} \in \mathbf{R}_{> 0}$', r'$C_{ij} \in \mathbf{R}$']

        fig = plt.figure()
        fig.set_size_inches(9, 5)
        fig.tight_layout(pad=3.0)
        ax = fig.add_subplot(111)

        for num_ax, experiment, title in zip(range(len(experiments)), experiments, titles):
            file_name = 'experiment_num_shots_' + experiment + '.json'
            file_name = path.join(results_folder, file_name)

            with open(file_name) as file:
                results = json.load(file)

            n = 10
            num_shots = [128, 256, 512, 1024]

            costs = np.zeros((len(num_shots), 2 ** n))
            acosts = np.zeros((len(num_shots), 2 ** n))
            qcosts = np.zeros((len(num_shots), 2 ** n))

            for i, num_shot in enumerate(num_shots):
                costs[i] = np.mean(results[str(n)][str(num_shot)]['costs'], axis=0)
                acosts[i] = np.mean(results[str(n)][str(num_shot)]['acosts'], axis=0)
                qcosts[i] = np.mean(results[str(n)][str(num_shot)]['qcosts'], axis=0)

            min_costs = np.min(costs, axis=1, keepdims=True)
            max_costs = np.max(costs, axis=1, keepdims=True)

            normalize = lambda c: (c - min_costs) / (max_costs - min_costs)
            costs = normalize(costs)
            acosts = normalize(acosts)
            qcosts = normalize(qcosts)
            error_costs[experiment] = np.linalg.norm(acosts - qcosts, axis=1)

            ax.scatter(num_shots, np.mean(np.abs(acosts - qcosts), axis=1), label=title)

        ax.set_xlabel('Number of Shots')
        ax.set_ylabel('Measurement Error')
        plt.legend()
        if save:
            save_fig(fig, path=path.join('../', 'images/experiment_num_shots.pdf'))
        plt.show()
        return

    vis_experiment_coef_K()
    vis_experiment_num_shots()
    return


def vis_benchmark_mc(file_name, save=False):
    """
    Visualises the Benchmark anlysis of UQMaxCut with QAOA and D-Wave
    Results are saved in the form of a dictionary {n:
                                                    ratio:
                                                        {qaoa_gradient: [[simulation 1], ..., [simulation n]],
                                                         qaoa_black_box: [[simulation 1], ..., [simulation n]],
                                                         dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]},
                                                    index:
                                                        {qaoa_gradient: [[simulation 1], ..., [simulation n]],
                                                         qaoa_black_box: [[simulation 1], ..., [simulation n]],
                                                         dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]},
                                                    time:
                                                        {qaoa_gradient: [[simulation 1], ..., [simulation n]],
                                                         qaoa_black_box: [[simulation 1], ..., [simulation n]],
                                                         dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]}}
    """

    file_name = path.join(results_folder, file_name)
    ns = [3, 5, 10, ]

    with open(file_name) as file:
        results = json.load(file)

    def plot_benchmark_ratio(ax, title, qaoa_gradient, qaoa_black_box, dwave, mc):
        num_iter = qaoa_gradient.shape[1]
        qaoa_gradient_data = qaoa_gradient.tolist()
        qaoa_black_box_mean = np.mean(qaoa_black_box, axis=0)[-1]
        dwave_mean = np.mean(dwave, axis=0)
        mc_data = mc.tolist()

        ax.axhline(xmin=-1, xmax=num_iter, y=qaoa_black_box_mean, linewidth=linewidth, c=qaoa_black_box_color)
        ax.axhline(xmin=-1, xmax=num_iter, y=dwave_mean, linewidth=linewidth, c=dwave_color)
        ax.plot(np.mean(qaoa_gradient_data, axis=0), linewidth=linewidth, c=qaoa_gradient_color)
        ax.fill_between(list(range(num_iter)),
                        np.mean(qaoa_gradient_data, axis=0)-np.std(qaoa_gradient_data, axis=0),
                        np.mean(qaoa_gradient_data, axis=0)+np.std(qaoa_gradient_data, axis=0),
                        color=qaoa_gradient_color, alpha=.2)
        ax.plot(np.mean(mc_data, axis=0), linewidth=linewidth, c=mc_color)
        ax.fill_between(list(range(num_iter)),
                        np.mean(mc_data, axis=0)-np.std(mc_data, axis=0),
                        np.mean(mc_data, axis=0)+np.std(mc_data, axis=0),
                        color=mc_color, alpha=.2)

        idx = np.arange(0, num_iter, num_iter//10)
        ax.set_ylim([0, 1.1])
        ax.set_xticks(idx, idx)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_title(title)

    def plot_benchmark_index(ax, x_start, index_data, color):
        width = 1.5
        x = np.linspace(x_start, x_start+6, 3)
        ax.bar(x, index_data, width, color=color)

        ax.text(x[0], index_data[0] + .08, f'$n=3$\n${int(index_data[0]*100)}\%$', horizontalalignment='center')
        ax.text(x[1], index_data[1] + .08, f'$n=5$\n${int(index_data[1]*100)}\%$', horizontalalignment='center')
        ax.text(x[2], index_data[2] + .08, f'$n=10$\n${int(index_data[2]*100)}\%$', horizontalalignment='center')
        return

    fig, ax = plt.subplot_mosaic('ABC;DDD', gridspec_kw={'height_ratios': [1.5, 1]})
    fig.set_size_inches(12, 7)
    fig.tight_layout(pad=6, h_pad=2, w_pad=2)
    for ax_index, n in zip(['A', 'B', 'C'], [str(x) for x in ns]):
        plot_benchmark_ratio(ax[ax_index], title=rf'$n = {n}$',
                             qaoa_gradient=np.array(results[n]['ratio']['qaoa_gradient']),
                             qaoa_black_box=np.array(results[n]['ratio']['qaoa_black_box']),
                             dwave=np.array(results[n]['ratio']['dwave']),
                             mc=np.array(results[n]['ratio']['mc']))
    ax['A'].set_ylabel('Ratio')


    index_data = lambda method: [np.mean(np.array(results[str(ns[0])]['index'][method])[:, -1]),
                                 np.mean(np.array(results[str(ns[1])]['index'][method])[:, -1]),
                                 np.mean(np.array(results[str(ns[2])]['index'][method])[:, -1])]
    plot_benchmark_index(ax['D'], x_start=0, index_data=index_data('qaoa_gradient'), color=qaoa_gradient_color)
    plot_benchmark_index(ax['D'], x_start=10, index_data=index_data('qaoa_black_box'), color=qaoa_black_box_color)
    plot_benchmark_index(ax['D'], x_start=20, index_data=index_data('dwave'), color=dwave_color)
    plot_benchmark_index(ax['D'], x_start=30, index_data=index_data('mc'), color=mc_color)
    ax['D'].set_xticks([3, 13, 23, 33], ['QAOA (NGD)', 'QAOA (Cobyla)', 'D-Wave', 'UQMaxCut (NGD) [Ours]'])
    ax['D'].set_yticks([0, .25, .50, .75, 1.], [0, 25, 50, 75, 100])
    ax['D'].set_ylim([0, 1.3])
    ax['D'].set_ylabel(f'Index $(\%)$')

    qaoa_gradient_patch = mpatches.Patch(color=qaoa_gradient_color, label=r'QAOA (NGD)')
    qaoa_black_box_patch = mlines.Line2D([], [], color=qaoa_black_box_color, label=r'QAOA (Cobyla)')
    dwave_patch = mlines.Line2D([], [], color=dwave_color, label=r'D-Wave')
    mc_patch = mpatches.Patch(color=mc_color, label=r'UQMaxCut (NGD) [Ours]')
    fig.legend(handles=[qaoa_gradient_patch, qaoa_black_box_patch, dwave_patch, mc_patch], loc='upper center', ncol=4, fontsize=font_size)

    if save:
        save_fig(fig, path=path.join('../', 'images/experiment_benchmark_mc.pdf'))
    fig.suptitle(file_name, y=.08)
    plt.show()
    return


def vis_benchmark_ising(file_name, save=False):
    """
    Visualises the Benchmark anlysis of UQMaxCut with QAOA and D-Wave
    Results are saved in the form of a dictionary {n:
                                                    ratio:
                                                        {dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]},
                                                    index:
                                                        {dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]},
                                                    time:
                                                        {dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]}}
    """

    file_name = path.join(results_folder, file_name)
    ns = [3, 5, 10, ]

    with open(file_name) as file:
        results = json.load(file)

    def plot_benchmark_ratio(ax, title, dwave, mc):
        num_iter = mc.shape[1]
        dwave_mean = np.mean(dwave, axis=0)
        mc_data = mc.tolist()

        ax.axhline(xmin=-1, xmax=num_iter, y=dwave_mean, linewidth=linewidth, c=dwave_color)
        ax.plot(np.mean(mc_data, axis=0), linewidth=linewidth, c=mc_color)
        ax.fill_between(list(range(num_iter)),
                        np.mean(mc_data, axis=0)-np.std(mc_data, axis=0),
                        np.mean(mc_data, axis=0)+np.std(mc_data, axis=0),
                        color=mc_color, alpha=.2)

        idx = np.arange(0, num_iter, num_iter//10)
        ax.set_ylim([0, 1.1])
        ax.set_xticks(idx, idx)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_title(title)

    def plot_benchmark_index(ax, x_start, index_data, color):
        width = 1.5
        x = np.linspace(x_start, x_start+6, 3)
        ax.bar(x, index_data, width, color=color)

        ax.text(x[0], index_data[0] + .08, f'$n=3$\n${int(index_data[0]*100)}\%$', horizontalalignment='center')
        ax.text(x[1], index_data[1] + .08, f'$n=5$\n${int(index_data[1]*100)}\%$', horizontalalignment='center')
        ax.text(x[2], index_data[2] + .08, f'$n=10$\n${int(index_data[2]*100)}\%$', horizontalalignment='center')
        return

    fig, ax = plt.subplot_mosaic('ABC;DDD', gridspec_kw={'height_ratios': [1.5, 1]})
    fig.set_size_inches(12, 7)
    fig.tight_layout(pad=6, h_pad=2, w_pad=2)
    for ax_index, n in zip(['A', 'B', 'C'], [str(x) for x in ns]):
        plot_benchmark_ratio(ax[ax_index], title=rf'$n = {n}$',
                             dwave=np.array(results[n]['ratio']['dwave']),
                             mc=np.array(results[n]['ratio']['mc']))
    ax['A'].set_ylabel('Ratio')


    index_data = lambda method: [np.mean(np.array(results[str(ns[0])]['index'][method])[:, -1]),
                                 np.mean(np.array(results[str(ns[1])]['index'][method])[:, -1]),
                                 np.mean(np.array(results[str(ns[2])]['index'][method])[:, -1])]
    plot_benchmark_index(ax['D'], x_start=10, index_data=index_data('dwave'), color=dwave_color)
    plot_benchmark_index(ax['D'], x_start=25, index_data=index_data('mc'), color=mc_color)
    ax['D'].set_xticks([13, 28], ['D-Wave', 'UQIsing (NGD) [Ours]'])
    ax['D'].set_yticks([0, .25, .50, .75, 1.], [0, 25, 50, 75, 100])
    ax['D'].set_xlim([3, 38])
    ax['D'].set_ylim([0, 1.3])
    ax['D'].set_ylabel(f'Index $(\%)$')

    dwave_patch = mlines.Line2D([], [], color=dwave_color, label=r'D-Wave')
    mc_patch = mpatches.Patch(color=mc_color, label=r'UQIsing (NGD) [Ours]')
    fig.legend(handles=[dwave_patch, mc_patch], loc='upper center', ncol=4, fontsize=font_size)

    if save:
        save_fig(fig, path=path.join('../', 'images/experiment_benchmark_ising.pdf'))
    fig.suptitle(file_name, y=.08)
    plt.show()
    return





if __name__ == '__main__':
    # Visualize the brute force experiment
    # vis_eval_mc(save=True)

    # Visualize the MaxCut benchmark experiment
    vis_benchmark_mc(file_name='experiment_benchmark_mc', save=False)

    # Visualize the Ising benchmark experiment
    vis_benchmark_ising(file_name='experiment_benchmark_ising', save=False)



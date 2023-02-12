import numpy as np
import matplotlib.pyplot as plt


def graph_convert(file_path):
    """
    将txt文件里的因果图转换为因果图邻接矩阵
    :param file_path: 文件路径
    :return: 邻接矩阵
    """
    f = open(file_path, "r")
    file_content = []
    for line in f:
        file_content.append(line.strip())

    f.close()
    # 确定因果变量数目
    causal_point = str(file_content[1])
    causal_point = len(causal_point.split(";"))

    # 确定关系语句位置
    file_content = file_content[4:-1]
    causal_relationship = []
    # 删除无关字符
    for str1 in file_content:
        str1 = str1.split(". ")[1]
        str1 = str1.split(" --> ")
        str1[0] = str1[0][1]
        str1[1] = str1[1][1]
        causal_relationship.append(str1)
    # print(causal_relationship)

    # 创建因果邻接矩阵
    true_graph = np.mat(np.zeros((causal_point, causal_point)))
    for one in causal_relationship:
        true_graph[int(one[0]) - 1, int(one[1]) - 1] = 1

    return np.asarray(true_graph)


class output_adj(object):
    """
    Visualization for causal discovery learning results.

    Parameters
    ----------
    est_dag: np.ndarray
        The DAG matrix to be estimated.
    true_dag: np.ndarray
        The true DAG matrix.
    show: bool
        Select whether to display pictures.
    save_name: str
        The file name of the image to be saved.
    """

    def __init__(self, est_dag, true_dag=None, show=True, save_name=None):

        self.est_dag = est_dag
        self.true_dag = true_dag
        self.show = show
        self.save_name = save_name

        if not isinstance(est_dag, np.ndarray):
            raise TypeError("Input est_dag is not numpy.ndarray!")

        if true_dag is not None and not isinstance(true_dag, np.ndarray):
            raise TypeError("Input true_dag is not numpy.ndarray!")

        if not show and save_name is None:
            raise ValueError('Neither display nor save the picture! ' + \
                             'Please modify the parameter show or save_name.')

        output_adj._plot_dag(self.est_dag, self.true_dag, self.show, self.save_name)

    @staticmethod
    def _plot_dag(est_dag, true_dag, show=True, save_name=None):
        """
        Plot the estimated DAG and the true DAG.

        Parameters
        ----------
        est_dag: np.ndarray
            The DAG matrix to be estimated.
        true_dag: np.ndarray
            The True DAG matrix.
        show: bool
            Select whether to display pictures.
        save_name: str
            The file name of the image to be saved.
        """

        if isinstance(true_dag, np.ndarray):

            # trans diagonal element into 0
            for i in range(len(true_dag)):
                if est_dag[i][i] == 1:
                    est_dag[i][i] = 0
                if true_dag[i][i] == 1:
                    true_dag[i][i] = 0

            # set plot size
            fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)

            ax1.set_title('test_graph')
            map1 = ax1.imshow(est_dag, cmap='Greys', interpolation='none')
            ax1.set_xticks(range(0, est_dag.shape[0], 1), range(1, est_dag.shape[0] + 1, 1))
            ax1.set_yticks(range(0, est_dag.shape[0], 1), range(1, est_dag.shape[0] + 1, 1))
            # fig.colorbar(map1, ax=ax1)

            ax2.set_title('true_graph')
            map2 = ax2.imshow(true_dag, cmap='Greys', interpolation='none')
            ax2.set_xticks(range(0, true_dag.shape[0], 1), range(1, true_dag.shape[0] + 1, 1))
            ax2.set_yticks(range(0, true_dag.shape[0], 1), range(1, true_dag.shape[0] + 1, 1))
            # fig.colorbar(map2, ax=ax2)

            if save_name is not None:
                fig.savefig(save_name)
            if show:
                plt.show()

        elif isinstance(est_dag, np.ndarray):

            # trans diagonal element into 0
            for i in range(len(est_dag)):
                if est_dag[i][i] == 1:
                    est_dag[i][i] = 0

            # set plot size
            fig, ax1 = plt.subplots(figsize=(4, 3), ncols=1)

            ax1.set_title('test_graph')
            map1 = ax1.imshow(est_dag, cmap='Greys', interpolation='none')
            ax1.set_xticks(range(0, est_dag.shape[0], 1), range(1, est_dag.shape[0] + 1, 1))
            ax1.set_yticks(range(0, est_dag.shape[0], 1), range(1, est_dag.shape[0] + 1, 1))
            # fig.colorbar(map1, ax=ax1)

            if save_name is not None:
                fig.savefig(save_name)
            if show:
                plt.show()



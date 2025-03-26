import time
import numpy as np
from typing import List, Tuple
from numpy import loadtxt, sqrt
from numpy import arange, argsort, argwhere, empty, full, inf, intersect1d, max, ndarray, sort, sum, zeros
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import KDTree
import math
import csv
from numpy import ndarray, empty, zeros, argsort, intersect1d, sum
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix,fowlkes_mallows_score,normalized_mutual_info_score


class Natural_Neighbor(object):

    def __init__(self):
        self.nan_edges = {}  # Grafo dos vizinhos mutuos
        self.nan_num = {}  # Numero de vizinhos naturais de cada instancia
        self.repeat = {}  # Estrututa de dados que contabiliza a repeticao do metodo count
        self.target = []  # Conjunto das classes
        self.data = []  # Conjunto de instancias
        self.knn = {}  # Estrutura que armazena os vizinhos de cada instanica
        self.natural_neighbors = {}  # 自然邻居的索引
        self.clusters = ()  # 第一次获得的簇
        self.shared_neighbor = {}
        self.num_shared_neighbor = {}
        self.k = None
        self.remaining_points = None
        self.nc = None
        self.T = None

    # Divide o dataset em atributos e classes
    def load(self, filename):
        aux = []
        with open(filename, 'r') as dataset:
            data = list(csv.reader(dataset))
            for inst in data:
                if not inst:  # 检查inst是否为空
                    continue
                try:
                    row = [float(x) for x in inst]
                    self.target.append(row[-1])  # 假设最后一个元素是目标值
                    aux.append(row[:-1])  # 假设除了目标值以外的所有元素都是特征值
                except ValueError as e:
                    print("Error converting string to float:", e)
        self.data = np.array(aux)

    def read(self, trainX, trainY=None):
        self.target = np.array(trainY)
        self.data = np.array(trainX)

    def asserts(self):
        self.nan_edges = set()
        for j in range(len(self.data)):
            self.knn[j] = set()
            self.nan_num[j] = 0
            self.repeat[j] = 0

    # Retorna o numero de instancias que nao possuiem vizinho natural
    def count(self):
        nan_zeros = 0
        for x in self.nan_num:
            if self.nan_num[x] == 0:
                nan_zeros += 1
        return nan_zeros

    # Retorna os indices dos vizinhos mais proximos
    def findKNN(self, inst, r, tree):
        _, ind = tree.query([inst], r + 1)
        return np.delete(ind[0], 0)

    # Retorna o NaNe
    def algorithm(self, nc):
        self.nc = nc
        # ASSERT
        tree = KDTree(self.data)
        self.asserts()
        flag = 0
        r = 1

        while (flag == 0):
            for i in range(len(self.data)):
                knn = self.findKNN(self.data[i], r, tree)
                n = knn[-1]
                self.knn[i].add(n)
                if (i in self.knn[n] and (i, n) not in self.nan_edges):
                    self.nan_edges.add((i, n))
                    self.nan_edges.add((n, i))
                    self.nan_num[i] += 1
                    self.nan_num[n] += 1
                    # Update natural neighbors
                    self.natural_neighbors.setdefault(i, set()).add(n)
                    self.natural_neighbors.setdefault(n, set()).add(i)

            cnt = self.count()
            rep = self.repeat[cnt]
            self.repeat[cnt] += 1
            if cnt == 0 or rep >= math.sqrt(r - rep):
                graph = nx.Graph()
                graph.add_edges_from(self.nan_edges)
                cluster34s = list(nx.connected_components(graph))
                if len(cluster34s) <= nc:  # 添加条件判断
                    flag = 1
                else:
                    r += 1
            else:
                r += 1  # 在原有的条件下增加 r 的值
        # print("nan=", r)
        self.k = r
        return self.k, self.nan_edges, self.natural_neighbors, self.knn, self.nan_num

    def cluster_result(self):
        # 创建图并添加边
        graph = nx.Graph()

        # 聚类循环直到聚类数量符合要求
        while True:
            self.algorithm(self.nc)
            print(self.nc)
            # 添加边到图中
            graph.add_edges_from(self.nan_edges)
            # 使用连通子图进行聚类
            clusters = list(nx.connected_components(graph))

            # 判断聚类数量是否符合要求
            if len(clusters) <= self.nc:
                break
        # 只将原始数据集划分为了一个簇，全为红色
        # colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']  # 可以添加更多颜色
        # for i, cluster in enumerate(clusters):
        #     print(i,cluster)
        #     cluster_points = self.data[list(cluster)]
        #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i % len(colors)], label=f'Cluster {i}')
        # plt.show()

    def snan(self):
        self.cluster_result()
        # ---------------------------------------获取最近邻居索引---------------------------------------
        self.shared_neighbors = {}  # 用于存储每对点的共享邻居索引
        self.num_shared_neighbors = {}  # 用于存储每对点的共享邻居数量

        for key, neighbors_key_one in self.natural_neighbors.items():
            self.shared_neighbors[key] = {}
            self.num_shared_neighbors[key] = {}

            for neighbor_key_two in neighbors_key_one:
                self.shared_neighbors[key][neighbor_key_two] = set(neighbors_key_one) & set(
                    self.natural_neighbors[neighbor_key_two])
                self.num_shared_neighbors[key][neighbor_key_two] = len(self.shared_neighbors[key][neighbor_key_two])
        # print(self.shared_neighbors)
        # print(len(self.num_shared_neighbors),self.num_shared_neighbors)
        # self.draw()

        # print(self.num_shared_neighbors)
        edges = []  # 把边拿到#集合
        edge_weights = {}  # 用于存储边的权重,权重是每一对边的共享邻居数量
        # 算法的核心自适应阈值 论文发表后会公布后续代码
        # self.T = self.calculate_adaptive_thresholdT(key, neighbor)
        self.T = 20
        for key, neighbors_key_three in self.shared_neighbors.items():
            for neighbor, shared in neighbors_key_three.items():
                if shared and self.num_shared_neighbors[key][neighbor] > self.T:
                    edge_weight = len(shared)
                    if edge_weight > 0:
                        edges.append((key, neighbor))
                        # 存储边的权重到字典中
                        edge_weights[(key, neighbor)] = edge_weight

        # 检查是否有满足条件的边
        if not edges:
            edges = self.nan_edges  # 如果没有边满足条件，则将其设为nan_edges

        # Step 3: Create clusters using the edges
        graph = nx.Graph()
        graph.add_edges_from(edges)
        clusters = list(nx.connected_components(graph))  # 获得簇#列表
        # print("旧的", type(clusters), len(clusters), clusters)
        # print("旧的", type(clusters), len(clusters))
        # print("snan,dt:", T)

        all_points = set(range(len(self.data)))
        cluster_points = set()
        for cluster in clusters:
            cluster_points.update(cluster)
        self.remaining_points = list(all_points - cluster_points)  # 列表

        # print(len(self.remaining_points))

        remaining_points_set = set(self.remaining_points)
        edges_to_remove = set()

        for edge in self.nan_edges:
            if edge[0] in remaining_points_set or edge[1] in remaining_points_set:
                edges_to_remove.add(edge)

        self.nan_edges.difference_update(edges_to_remove)
        graph = nx.Graph()
        graph.add_edges_from(self.nan_edges)
        self.clusters = list(nx.connected_components(graph))  # 获得簇


        paper_cluster = {}  # 用于跟踪簇以避免问题的字典
        for i, va in enumerate(self.clusters):
            paper_cluster[i] = va


        # colormap = plt.cm.get_cmap('tab10')  # 已经被弃用了
        colormap = plt.colormaps['tab10']
        num_clusters = len(self.clusters)

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)  # 调整边距

        # 将索引扩展到20个颜色
        colors = [colormap(i % colormap.N) for i in range(num_clusters)]
        if num_clusters > 10:
            colormap = plt.cm.get_cmap('tab20')
            colors = [colormap(i % colormap.N) for i in range(num_clusters)]

        for i, clustery in enumerate(self.clusters):
            cluster_points = self.data[list(clustery)]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i % len(colors)], s=4)
            # plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color='black', s=2)

            # # 添加标签
            # for point_index in clustery:
            #     point = self.data[point_index]
            #     plt.annotate(str(point_index), (point[0], point[1]), xytext=(5, -5), textcoords='offset points',
            #                  fontsize=8, color='black')

        lines = []
        for edge in self.nan_edges:
            x = [self.data[edge[0], 0], self.data[edge[1], 0]]
            y = [self.data[edge[0], 1], self.data[edge[1], 1]]
            lines.append(list(zip(x, y)))

        edge_collection = LineCollection(lines, colors='gray', linestyle='-', alpha=0.6, linewidth=0.6)
        ax.add_collection(edge_collection)

        if len(self.remaining_points) > 0:
            remaining_points_data = self.data[self.remaining_points]
            plt.scatter(remaining_points_data[:, 0], remaining_points_data[:, 1],
                        c=['red'] * len(self.remaining_points), marker='x', s=1.5, edgecolors='b')

        # 3. 保存图像为 PDF、EPS 和 PNG 格式
        # file_name_no_ext = 'HCAPNG-xclara'
        # save_dir = r'C:\Users\Desktop'

        # # 保存为 PDF
        # plt.savefig(f'{save_dir}\\{file_name_no_ext}.pdf', format='pdf')
        # # 保存为 EPS
        # plt.savefig(f'{save_dir}\\{file_name_no_ext}.eps', format='eps')
        # # 保存为 PNG
        # plt.savefig(f'{save_dir}\\{file_name_no_ext}.png', format='png')
        #
        # plt.legend()
        plt.show()
        return paper_cluster

    def calculate_adaptive_thresholdT(self, x, y):

        T = 10
        return T

    def merge(self):
        cluster_IDs = [] # 需要合并的簇a和簇b的id
        paper_cluster = {}  # 做一个簇用与跟踪合并,防止有问题
        for i, va in enumerate(self.clusters):
            paper_cluster[i] = va
        while len(paper_cluster) > self.nc:  # 当合并后的簇数量大于阈值时继续合并
            sim = self.diance(paper_cluster)
            max_similarity = -1  # 最大相似度初始化为负数
            selected_pair = None

            for pair, similarity in sim.items():
                if similarity > max_similarity:  # 找到最相似的簇对
                    max_similarity = similarity
                    selected_pair = pair
            if selected_pair:
                cluster_a, cluster_b = selected_pair
                paper_cluster[cluster_a] = paper_cluster[cluster_b].union(paper_cluster[cluster_a])
                del paper_cluster[cluster_b]  # 删除被合并的

            new_paper_cluster = {}
            new_cluster_index = 0
            for old_cluster_index, papers in paper_cluster.items():
                new_paper_cluster[new_cluster_index] = papers
                new_cluster_index += 1
            paper_cluster = new_paper_cluster
        # self.plt(paper_cluster)
        paper_cluster = self.assign_remaining_points(paper_cluster)

        return paper_cluster

    # 分配剩余点
    def assign_remaining_points(self, paper_cluster):
        # 构建KD树
        kdtree = KDTree(self.data)

        for point in self.remaining_points:
            nearest_point_index = None
            found_cluster = False
            # 使用KD树寻找最近邻点
            _, nearest_point_indices = kdtree.query([self.data[point]], k=len(self.data))

            # 寻找最近邻点所属的簇
            for nearest_index in nearest_point_indices[0]:
                for cluster_idx, cluster in paper_cluster.items():
                    if nearest_index in cluster:
                        cluster.add(point)
                        found_cluster = True
                        break
                if found_cluster:
                    break

            # 如果未找到所属的簇，则继续寻找下一个最近邻点
            if not found_cluster:
                continue

        # 移除已分配的未分配点
        self.remaining_points = []
        # self.plt(paper_cluster)

        return paper_cluster


    def acc(self, paper_cluster):

        labels = np.zeros(len(self.data), dtype=int)  # 创建一个与数据点数量相同的标签数组，初始值为0
        for cluster_idx, cluster in paper_cluster.items():
            labels[list(cluster)] = cluster_idx + 1  # 将聚类结果填充到标签数组中，从1开始作为标签

        label_encoder = LabelEncoder()
        self.target = label_encoder.fit_transform(self.target)
        labels, self.target = ACC.matchY(labels, self.target)
        ON = ACC.measures_calculator(labels, self.target)


if __name__ == '__main__':
    # 添加本地路径
    path = r'C:\DM_Date\code\data\my data\pathbased.csv'  # 3

    start_time_algorithm = time.time()  # 记录算法开始时间
    nan = Natural_Neighbor()
    nan.load(path)
    nc = 7
    k, nan_edges, natural_neighbors, knn, nan_num = nan.algorithm(nc)
    paper_cluster = nan.snan()
    # paper_cluster = nan.merge(nc)  # 自适应+合并 （本文创新，论文发表后会更新后续代码）

    nan.acc(paper_cluster)
    end_time_algorithm = time.time()  # 记录算法结束时间
    runtime_algorithm = end_time_algorithm - start_time_algorithm  # 计算算法运行时间
    print("Total algorithm runtime:", runtime_algorithm)  # 打印算法总运行时间
    # # 计算评估指标
    # acc = cluster_accuracy(y_true, clusterer.labels_)
    # ami = adjusted_mutual_info_score(y_true, clusterer.labels_)
    # ari = adjusted_rand_score(y_true, clusterer.labels_)
    # fmi = fowlkes_mallows_score(y_true, clusterer.labels_)  # 新增FMI
    # nmi = normalized_mutual_info_score(y_true, clusterer.labels_)  # 新增NMI
    #
    # # # 输出结果
    # print(f"ACC: {acc:.4f}", f"AMI: {ami:.4f}", f"ARI: {ari:.4f}", f"FMI:{fmi:.4f}", f"NMI:{nmi:.4f}")
import os
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(17)

def main():
    # Setting up file paths
    dir = os.path.dirname(__file__)
    data_file_names = ["hip1000.txt", "pfc1000.txt", "str1000.txt"]
    gene_file_names = ["hip1000names.txt", "pfc1000names.txt", "str1000names.txt"]
    test_means_file_names = "test_mean.txt"
    # Reading in data
    test_means = ReadMeans(os.path.join(dir,test_means_file_names))
    mouse_data = ReadData(os.path.join(dir, "mouse-data", data_file_names[0]))
    mouse_data_genes = ReadGeneNames(os.path.join(dir, "mouse-data", gene_file_names[0]))

    PartA(mouse_data, mouse_data_genes, test_means)
    # PartB(mouse_data, mouse_data_genes)
    # PartC(mouse_data, mouse_data_genes)
    # PartD(mouse_data, mouse_data_genes)
     
    
def PartA(mouse_data: np.ndarray, mouse_data_genes: list[str], test_means: np.ndarray):
    groups,loss = KMeans(mouse_data, mouse_data_genes, 3, test_means)
    save_path = os.path.join(os.path.dirname(__file__), "images\\4a.png")
    plot_title = f"Loss vs Training Generation for k = {3}"
    PlotObjectiveFunction(loss, 3, plot_title, save_path)
    print(f"Loss at convergence {loss[-1]}")

def PartB(mouse_data: np.ndarray, mouse_data_genes: list[str]):
    raw_corr_coeff = np.corrcoef(mouse_data, rowvar=False)
    fig = plt.figure(1)
    ax = fig.subplots(1,2)
    ax[0].imshow(raw_corr_coeff)
    ax[0].set_title("Raw Correlation Matrix")

    groups, loss = KMeans(mouse_data, mouse_data_genes, 3)
    save_path = os.path.join(os.path.dirname(__file__), "Images\\4b.png")

    rearranged_data = GetRearrangedData(groups)
    new_corr_coeff = np.corrcoef(rearranged_data, rowvar = False)

    ax[1].imshow(new_corr_coeff)
    ax[1].set_title("Clustered Correlation Matrix")
    
    fig.savefig(save_path)

def PartC(mouse_data: np.ndarray, mouse_data_genes: list[str]):
    save_path = os.path.join(os.path.dirname(__file__), "Images\\4c.png")

    fig = plt.figure()
    axs = fig.subplots(2, 5)
    for i in range(10):
        groups, loss = KMeans(mouse_data, mouse_data_genes, 3)
        rearranged_data = GetRearrangedData(groups)
        corr_coef = np.corrcoef(rearranged_data, rowvar = False)
        axs[int(np.floor(i/5)), i%5].imshow(corr_coef)
        axs[int(np.floor(i/5)), i%5].set_title(f"Loss = {loss[-1]:3.2e}", fontsize=6)   

    fig.tight_layout(w_pad = 0.25)
    fig.suptitle("10 Random Starts. k = 3")
    fig.savefig(save_path)


def PartD(mouse_data: np.ndarray, mouse_data_genes: list[str]):
    y = {}
    for idx, val in enumerate(range(3,13)):
        best_loss = np.inf
        for idx1 in range(10):
            groups = KMeans(mouse_data, mouse_data_genes, val)
            if groups[1][-1] < best_loss:
                best_loss = groups[1][-1]
                y[val] = groups

    fig = plt.figure()
    axs = fig.subplots(2, 5) 
    for idx, k_size in enumerate(y):
        rearranged_data = GetRearrangedData(y[k_size][0])
        corr_coef = np.corrcoef(rearranged_data, rowvar = False)
        axs[int(np.floor(idx/5)), idx%5].imshow(corr_coef)
        axs[int(np.floor(idx/5)), idx%5].set_title(f"K = {k_size}\nLoss = {y[k_size][1][-1]: 3.2e}", fontsize = 6)

    fig.tight_layout(w_pad = 0.25)
    fig.suptitle("Correlation Matrices for k=3 to k=12")
    save_path = os.path.join(os.path.dirname(__file__), "Images\\4d.png")
    fig.savefig(save_path)
    plt.close(fig)

    x = range(3, 13)    
    losses = [0 for i in range(3, 13)]
    for idx, key in enumerate(y):
        losses[idx] = y[key][1][-1]
    
    save_path = os.path.join(os.path.dirname(__file__), "Images\\4d2.png")
    fig = plt.figure()
    axs = fig.subplots()
    axs.plot(x, losses)
    axs.scatter(x, losses)
    axs.set_xticks(range(3,13))
    axs.set_title("K vs Final Loss")
    axs.set_ylabel("Loss")
    axs.set_xlabel("K")
    fig.savefig(save_path)

def ReadMeans(path: str) -> np.ndarray:
    """
    Reads the test input means.
    """
    means = np.loadtxt(path)
    return means

def ReadData(path: str) -> np.ndarray:
    """
    Reads the test data. 
    """
    rawdata = np.loadtxt(path, delimiter=",")
    return rawdata

def ReadGeneNames(path: str) -> np.ndarray:
    with open(path) as f:
        gene_names = f.readline().strip().split(",")
    return gene_names

def KMeans(x: np.ndarray, names: list[str], k: int,
            means: np.ndarray = np.array(0)) -> dict[int, list[np.ndarray, np.ndarray, list[str], list[int]]]:
    """
    Performs KMeans Clustering.
    input:
        x: The data set.
        names: The names of each ponit in the data.
        means: The means of the data
        k: The number of means to use
    """
    # Generate the random starting means if starting means not given.
    if len(means.shape) == 0:
        means = np.zeros((x.shape[0], k))
        mean_cols = rng.choice(x.shape[1], size = k, replace = False)
        for idx, col in enumerate(mean_cols):
            means[:,idx] = x[:, col]

    current_cluster_index = [0 for i in range(x.shape[1])]
    loss = []
    iterations = 0
    while True:
        # print(f"\rNum Iterations: {iterations}", end = "")
        iterations += 1
        new_cluster_indexes = GenerateClusterIndexes(x, means, k)

        groups_idx_dict = MakeGroupDict(new_cluster_indexes, k)
        groups = MakeKMeansDict(means, x, groups_idx_dict, names)
        loss.append(CalculateObjectiveFunction(groups))
        
        if new_cluster_indexes == current_cluster_index:
            current_cluster_index = new_cluster_indexes
            break
        current_cluster_index = new_cluster_indexes
        means = CalculateNewMeans(x, current_cluster_index, k, means)

    groups_idx_dict = MakeGroupDict(current_cluster_index, k)
    groups = MakeKMeansDict(means, x, groups_idx_dict, names)
   
    return groups, loss

def GetRearrangedData(groups: dict[int, list[np.ndarray, np.ndarray, list[str]]]) -> np.ndarray:
    """
    GetRearrangedData: Takes the clusters of data and concatentates all the data points into one large
    matrix.
    Input:
        groups: The clustered data.
            key = cluster
            value = list[means, data from cluster, gene names]
    Output:
        The concatenated data as an np.ndarray.
    """
    num_rows = groups[0][1].shape[0]
    num_cols = 0
    for key in groups:
        num_cols += groups[key][1].shape[1]

    start_idx = 0
    rearranged_data = np.zeros((num_rows, num_cols))
    for key in groups:
        num_samples = groups[key][1].shape[1]
        end_idx = start_idx + num_samples
        rearranged_data[:, start_idx:end_idx] = groups[key][1]
        start_idx = end_idx 

    return rearranged_data

def PlotObjectiveFunction(loss: list[int], k: int, title: str, savepath: str):
    """
    PlotObjectiveFunction: Plots and saves the objective function into a png file.
    Input:
        loss: The loss function.
        k: number of means
    """
    fig = plt.figure(k)
    axs = fig.subplots()
    axs.plot(loss)
    axs.scatter(range(len(loss)), loss)
    axs.set_title(title)
    axs.set_ylabel("Loss")
    axs.set_xlabel("Iteration")
    fig.savefig(savepath)
    fig.clf()
    

def MakeKMeansDict(means: np.ndarray, x: np.ndarray, groups_idx: list[int],
                    names: list[str]) -> dict[int, list[np.ndarray, np.ndarray, list[str]]]:
    """
    MakeKMeansDict: MakeKMeansDict makes a dictionary for k means grouping.
    Input:
        means: The means of the clusters.
        x: The data.
        group_idx: A dictionary where k = the group and value = the index of the row in X that belongs to the group.
        names: The names of the data points in X.
    """
    groups = {}
    for key in groups_idx:
        group_data = x[:, groups_idx[key]]
        group_data_names = [names[idx] for idx in groups_idx[key]]
        groups[key] = [means[:, key], group_data, group_data_names]
    return groups

def GenerateClusterIndexes(x: np.ndarray, means: np.ndarray, k: int) -> list[int]:
    """
    GenerateClusters: Generates the clusters of points around each mean.
    x: All of the data stored as an numpy array. Each column is a data point.
    names: The list of names corresponding to each data point.
    means: The current means. Each column is a set of coordinates to describe a point.
    k: The number of means to use.
    """
    groups = [0 for i in range(x.shape[1])]
    for idx0 in range(x.shape[1]):
        groups[idx0] = ChooseCluster(x[:,idx0], means, k)
    return groups

def ChooseCluster(x: np.ndarray, means: np.ndarray, k: int) -> int:
    """
    Choose the cluster that a single point should belong to.
    Input:
        x: An array representing the point being evaluated
        means: The center of the clusters.
        k: The number of clusters
    Output:
        The index of the cluster in means.
    """
    best_idx = 0
    best_dist = np.inf
    for idx in range(k):
        distance = EuclideanDistance(x, means[:,idx])
        if distance < best_dist:
            best_dist = distance
            best_idx = idx
    return best_idx
    

def CalculateNewMeans(x: np.ndarray, group_idx: list[int], k: int, old_means: np.ndarray):
    """
    Generate the new means of each cluster.
    Input:
        x: The data set.
        group_idx: The group each data point belongs to.
        k: The number of means to calculate.
    """
    new_means = old_means.copy()

    groups = MakeGroupDict(group_idx, k)
    for idx in range(k):
        if len(groups[idx]) > 0:
            cluster_data = x[:, groups[idx]]
            new_means[:, idx] = CalculateMean(cluster_data)
    
    return new_means

def CalculateMean(x: np.ndarray):
    """
    Calculates the mean of a set of points.
    Input:
        x: Each column is one data point. The number of rows is the number of dimensions.
    Output:
        The mean of the data points.
    """
    return np.mean(x, axis = 1)
    
def MakeGroupDict(group_idx: list[int], k: int) -> dict[int, list[int]]:
    """
    MakeGroupDict: Takes a list of values corresponding to what group an index belongs to. Creates
    a dictionary so that the key = the group and the values = the indicies of the members in that group.
    Input:
        group_idx: The list of which group an index belongs in.
        k: number of groups.
    """
    groups = {}
    for idx in range(k): # Setting up a dictionary for easy access to indexes in each group.
        groups[idx] = []

    for idx in range(len(group_idx)): # Appending the indexes to their propper group.
        index = group_idx[idx]
        groups[index].append(idx)
    
    return groups

def CalculateObjectiveFunction(groups_dict: dict[int, list[np.ndarray, np.ndarray, list[str]]]) -> float:
    """
    CalculateObjectiveFunction: Calculates the objective function score of a Kmeans grouping. 
    Input:
        Groups_dict: dictionary for keeping track of the groups.
    """
    sum_distances = 0
    for key in groups_dict: # For each group.
        if key != "loss":
            data_points = groups_dict[key][1]
            center = groups_dict[key][0]
            for idx in range(data_points.shape[1]):
                sum_distances += EuclideanDistance(center, data_points[:, idx])
    return sum_distances

def EuclideanDistance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculates the distance between two points represented as np.ndarrays.
    Input:
        p1: First point
        p2: Second Point
    Output:
        The euclidian distance between p1 and p2.
    """
    delta = np.sum([p1, -1*p2], axis = 0)
    distance = np.linalg.norm(delta)
    return distance

if __name__ == main():
    main()
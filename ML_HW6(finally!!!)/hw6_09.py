import numpy as np
import requests

def parse_libsvm_line(line):
    elements = line.split()
    target = float(elements[0])
    features = np.zeros(8)  # 根据数据集特征数量调整
    for e in elements[1:]:
        index, value = e.split(":")
        features[int(index) - 1] = float(value)
    return features, target

def download_data(url):
    response = requests.get(url)
    data = response.content.decode('utf-8').splitlines()
    features, targets = zip(*[parse_libsvm_line(line) for line in data])
    return np.array(features), np.array(targets)

def split_dataset(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

def squared_error(y):
    return np.sum((y - np.mean(y)) ** 2)

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def get_potential_splits(X):
    potential_splits = {}
    _, n_features = X.shape
    for feature_index in range(n_features):
        values = X[:, feature_index]
        unique_values = np.unique(values)

        potential_splits[feature_index] = []
        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2
                potential_splits[feature_index].append(potential_split)
    
    return potential_splits

def build_tree(X, y):
    num_samples, num_features = X.shape

    # 如果所有的樣本都是同一類別，或只有一個樣本，則返回葉節點
    if num_samples == 1 or len(np.unique(y)) == 1:
        leaf_value = np.mean(y)
        return TreeNode(value=leaf_value)

    best_feature, best_threshold, best_error = None, None, float('inf')
    current_error = squared_error(y)
    potential_splits = get_potential_splits(X)

    for feature_index in potential_splits:
        for threshold in potential_splits[feature_index]:
            _, y_left, _, y_right = split_dataset(X, y, feature_index, threshold)
            error = squared_error(y_left) + squared_error(y_right)

            if error < best_error:
                best_error, best_feature, best_threshold = error, feature_index, threshold

    if best_feature is not None:
        X_left, y_left, X_right, y_right = split_dataset(X, y, best_feature, best_threshold)
        left_subtree = build_tree(X_left, y_left)
        right_subtree = build_tree(X_right, y_right)
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    # 如果無法進一步分割，返回葉節點
    return TreeNode(value=np.mean(y))

def predict(tree, X):
    if tree.value is not None:
        return tree.value
    feature_val = X[tree.feature]
    branch = tree.left if feature_val <= tree.threshold else tree.right
    return predict(branch, X)

def main(train_url, test_url):
    X_train, y_train = download_data(train_url)
    X_test, y_test = download_data(test_url)

    tree = build_tree(X_train, y_train)

    predictions = [predict(tree, x) for x in X_test]
    e_out = np.mean((y_test - predictions) ** 2)

    print(f'E_out: {e_out}')

# 執行
train_url = 'http://www.csie.ntu.edu.tw/~htlin/course/ml23fall/hw6/hw6_train.dat'
test_url = 'http://www.csie.ntu.edu.tw/~htlin/course/ml23fall/hw6/hw6_test.dat'
main(train_url, test_url)

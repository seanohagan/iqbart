import numpy as np
from typing import Callable

class Node:
    def __init__(self,
                 id : int,
                 feature_idx: int | None = None,
                 threshold: float | None = None,
                 left: int | None = None,
                 right: int | None = None,
                 parent: int | None = None,
                 depth : int = 0,
                 value: float | None = None):

        self.id = id
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.parent = parent
        self.value = value
        self.depth = depth

    def _is_leaf(self) -> bool:
        return self.left is None and self.right is None


class Tree:
    def __init__(self):
        self.nodes = {} #Id: node
        self.root = None
        self.next_id = 0

        # Create root node
        root = self.create_node(id=self.next_id, parent = None, depth=0, value=None)
        self.set_root(root)

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def set_root(self, node: Node):
        self.root = node

    def create_node(self, id: int, parent: int | None, depth: int, value: float | None) -> Node:
        node = Node(id=id, parent=parent, depth=depth, value=value)
        self.next_id += 1
        self.add_node(node)
        return node

    def split_node(self, node: Node, feature_idx: int, threshold: float) -> None:
        if not node._is_leaf():
            raise ValueError("Node is not a leaf")

        node.feature_idx = feature_idx
        node.threshold = threshold
        parent_id = node.id
        parent_depth = node.depth
        node.left = self.create_node(self.next_id, parent=parent_id, depth=parent_depth+1, value=None).id
        node.right = self.create_node(self.next_id, parent=parent_id, depth=parent_depth+1, value=None).id

    def _get_leaves(self) -> list[Node]:
        return [node for node in self.nodes.values() if node._is_leaf()]

    def _is_left_child(self, node: Node) -> bool:
        if node.parent is None:
            return False
        return node.id == self.nodes[node.parent].left

    def _is_right_child(self, node: Node) -> bool:
        if node.parent is None:
            return False
        return node.id == self.nodes[node.parent].right

def criterion_variance(y: np.ndarray) -> float:
    return float(np.var(y))

def criterion_quantile(y: np.ndarray, q: float = 0.5) -> float:
    yhat = np.quantile(y, q)
    loss = np.where(y - yhat < 0, -(1-q)*(y-yhat), q*(y-yhat))
    return float(np.mean(loss))
    # if y - yhat < 0:
    #     return -(1-q)*(y-yhat)
    # else:
    #     return q*(y-yhat)

def get_quantile_criterion(q: float) -> Callable:
    return lambda y: criterion_quantile(y, q)

def value_mean(y: np.ndarray) -> float:
    return float(np.mean(y))

def value_quantile(y: np.ndarray, q: float = 0.5, output_shape=1) -> float:
    return np.quantile(y, q)

class TreeRegressor:
    def __init__(self, domain: np.ndarray,
                 criterion_fn: Callable = criterion_variance,
                 value_fn : Callable = value_mean,
                 max_depth: int = 5,
                 min_decrease : float = 0.0):
        self.tree = Tree()
        self.is_fitted = False
        self.criterion_fn = criterion_fn
        self.value_fn = value_fn

        self.max_depth = max_depth
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.min_decrease = min_decrease
        self.n_splits_per_node_per_dim = 10

        self.X = None
        self.y = None

        self.domain = domain

    def calculate_criterion_reduction(self, y_node: np.ndarray,
                                      y_left: np.ndarray, y_right: np.ndarray,
                                      relative : bool = False) -> float:
        n_left = y_left.shape[0]
        n_right = y_right.shape[0]
        n_total = y_node.shape[0]



        if n_total != n_left + n_right:
            raise ValueError("Invalid split")

        criterion_pre_split = self.criterion_fn(y_node)
        criterion_post_split = (n_left / n_total) * self.criterion_fn(y_left) + (n_right / n_total) * self.criterion_fn(y_right)

        if criterion_pre_split == 0.0:
            return 0.0


        if relative:
            return (criterion_pre_split - criterion_post_split) / criterion_pre_split

        return criterion_pre_split - criterion_post_split

    def get_path_to_node(self, node: Node) -> list[int]:
        path = []
        current = node.id
        while current != self.tree.root.id:
            path.append(current)
            current = self.tree.nodes[current].parent
        path.append(self.tree.root.id)
        return list(reversed(path))


    def get_mask_in_node(self, node: Node) -> np.ndarray:

        path = self.get_path_to_node(node)
        current_mask = np.ones(self.X.shape[0], dtype=bool)
        for i, node_id in enumerate(path):
            if i == 0:
                mask = np.ones(self.y.shape[0], dtype=bool)
            else:
                parent = self.tree.nodes[node_id].parent
                feature_idx = self.tree.nodes[parent].feature_idx
                threshold = self.tree.nodes[parent].threshold
                if self.tree._is_left_child(self.tree.nodes[node_id]):
                    mask = self.X[:, feature_idx] < threshold
                elif self.tree._is_right_child(self.tree.nodes[node_id]):
                    mask = self.X[:, feature_idx] >= threshold
                else:
                    raise ValueError("Invalid node")
            current_mask = current_mask & mask

        return current_mask


    def calculate_split(self, node: Node, feature_idx: int, threshold: float,
                        relative: bool = False) -> float:

        node_mask = self.get_mask_in_node(node)

        X_node = self.X[node_mask]
        y_node = self.y[node_mask]

        mask = X_node[:, feature_idx] < threshold
        y_left = y_node[mask]
        y_right = y_node[~mask]
        return self.calculate_criterion_reduction(y_node, y_left, y_right, relative)

    def get_feature_constraints(self, node: Node, feature: int) -> np.ndarray:
        path = self.get_path_to_node(node)
        # print(f"Path: {path}")
        feature_lb = self.domain[feature, 0]
        feature_ub = self.domain[feature, 1]

        for i, node_id in enumerate(path):
            if i == 0:
                continue
            current_node = self.tree.nodes[node_id]
            current_parent = self.tree.nodes[current_node.parent]
            feature_idx = current_parent.feature_idx
            if feature_idx == feature:
                threshold = current_parent.threshold
                if self.tree._is_left_child(current_node):
                    feature_ub = threshold
                elif self.tree._is_right_child(current_node):
                    feature_lb = threshold
                else:
                    raise ValueError("Invalid node")

        return np.array([feature_lb, feature_ub])


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.is_fitted:
            raise ValueError("Model is already fitted")

        if not (X.ndim == 2):
            raise ValueError("X must be a 2D array of shape (n,d)")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if not (X.shape[1] > 0 and y.shape[0] > 0):
            raise ValueError("X and y cannot be empty")

        self.X = X
        self.y = y
        self.tree.nodes[self.tree.root.id].value = self.value_fn(y)

        can_continue = True

        while can_continue:
            leaves = self.tree._get_leaves()
            # print(f"Leaves: {[leaf.id for leaf in leaves]}")
            split_criteria = np.zeros((len(leaves), X.shape[1], self.n_splits_per_node_per_dim))
            thresholds = np.zeros((len(leaves), X.shape[1], self.n_splits_per_node_per_dim))
            for i,leaf in enumerate(leaves):
                for j,feature in enumerate(range(X.shape[1])):
                    thresholds[i,j] = self._get_split_candidates(leaf, feature)
                    for k,threshold in enumerate(thresholds[i,j]):
                        if self._is_valid_split(leaf, feature, threshold):
                            split_criteria[i,j,k] = self.calculate_split(leaf, feature, threshold)
                        else:
                            split_criteria[i,j,k] = -np.inf


            # print(f"Split criteria: {split_criteria}, shape: {split_criteria.shape}")

            best_split = np.unravel_index(np.argmax(split_criteria), split_criteria.shape)
            best_leaf = leaves[best_split[0]]
            best_feature = int(best_split[1])
            best_threshold = thresholds[best_split]

            # print(f"Best leaf: {best_leaf.id}. Threshold: {best_threshold}")

            # print(f"Best split: {best_split}. Criteria: {split_criteria[best_split]}")
            #

            if split_criteria[best_split] > -np.inf:
                relative_improvement = self.calculate_split(leaves[best_split[0]], best_split[1], thresholds[best_split], relative=True)
            else:
                relative_improvement = -np.inf

            if relative_improvement > self.min_decrease:
                self.tree.split_node(best_leaf, best_feature, best_threshold)
                left_child_id = self.tree.nodes[best_leaf.id].left
                right_child_id = self.tree.nodes[best_leaf.id].right
                self.tree.nodes[left_child_id].value = self.value_fn(self.y[self.get_mask_in_node(self.tree.nodes[left_child_id])])
                self.tree.nodes[right_child_id].value = self.value_fn(self.y[self.get_mask_in_node(self.tree.nodes[right_child_id])])

            else:
                can_continue = False

        self.is_fitted = True

    def _is_valid_split(self, node: Node, feature : int, threshold: float) -> bool:

        # print(f"Node depth: {node.depth}, max depth: {self.max_depth}")

        if node.depth >= self.max_depth:
            return False

        node_mask = self.get_mask_in_node(node)
        X_node = self.X[node_mask]
        # y_node = self.y[node_mask]

        left_mask = node_mask & (self.X[:, feature] < threshold)
        right_mask = node_mask & (self.X[:, feature] >= threshold)

        # print(f"Threshold: {threshold}")
        # X_left = X_node[X_node[:, feature] < threshold]
        # X_right = X_node[X_node[:, feature] >= threshold]

        X_left = self.X[left_mask]
        X_right = self.X[right_mask]

        # print(f"X_node: {X_node.shape[0]}, X_left: {X_left.shape[0]}. X_right: {X_right.shape[0]}")

        if X_node.shape[0] < self.min_samples_split:
            return False
        if X_left.shape[0] < self.min_samples_leaf or X_right.shape[0] < self.min_samples_leaf:
            return False

        return True

    def _get_split_candidates(self, node: Node, feature: int) -> np.ndarray:
        feature_lb, feature_ub = self.get_feature_constraints(node, feature)
        # print(f"Feature constraints: node: {node.id}, lb {feature_lb}, ub {feature_ub}")
        return np.linspace(feature_lb, feature_ub, self.n_splits_per_node_per_dim)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:

        leaf_indices = self.apply(X)
        if "output_shape" in kwargs:
            predictions = np.zeros((leaf_indices.shape[0], kwargs["output_shape"]))
        else:
            predictions = np.zeros(leaf_indices.shape[0])

        for i,li in enumerate(leaf_indices):
            mask = self.get_mask_in_node(self.tree.nodes[leaf_indices[i]])
            predictions[i] = self.value_fn(self.y[mask], **kwargs)

        return predictions


    def apply(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        if not (X.ndim == 2):
            raise ValueError("X must be a 2D array of shape (n,d)")

        if X.shape[1] != self.X.shape[1]:
            raise ValueError("X must have the same number of features as the training data")

        leaf_indices = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = self.tree.root
            while not node._is_leaf():
                if X[i, node.feature_idx] < node.threshold:
                    node = self.tree.nodes[node.left]
                else:
                    node = self.tree.nodes[node.right]
            leaf_indices[i] = node.id

        return leaf_indices


class RandomForest:
    def __init__(self, n_trees: int, domain: np.ndarray,
                 value_fn: Callable = value_mean,
                 min_decrease: float = 0.0,
                 max_depth: int = 5,
                 **kwargs):
        self.n_trees = n_trees
        self.domain = domain
        self.bootstrap = True
        self.max_features = 0.8
        self.min_decrease = min_decrease
        self.trees = [TreeRegressor(domain, value_fn=value_fn,
                                    min_decrease=min_decrease,
                                    max_depth=max_depth) for _ in range(n_trees)]
        self.tree_feature_sets = []

    def fit(self, X: np.ndarray, y: np.ndarray, PRNGKey: int | None = None) -> None:
        if PRNGKey is not None:
            np.random.seed(PRNGKey)

        for tree in self.trees:
            if self.bootstrap:
                idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
                X_sample = X[idx]
                y_sample = y[idx]
            else:
                X_sample = X
                y_sample = y
            if isinstance(self.max_features, float):
                features_sample = np.random.choice(X.shape[1], max(1,int(self.max_features * X.shape[1])), replace=False)
            elif isinstance(self.max_features, int):
                features_sample = np.random.choice(X.shape[1], self.max_features, replace=False)
            tree.fit(X_sample[:, features_sample], y_sample)
            self.tree_feature_sets.append(features_sample)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if "output_shape" in kwargs:
            predictions = np.zeros((X.shape[0], self.n_trees, kwargs["output_shape"]))
        else:
            predictions = np.zeros((X.shape[0], self.n_trees))
        for i, tree in enumerate(self.trees):
            X_fs = X[:, self.tree_feature_sets[i]]
            predictions[:, i] = tree.predict(X_fs, **kwargs)
        return np.mean(predictions, axis=1)


def fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                x_grid: np.ndarray, q_grid: np.ndarray,
                n_trees: int = 200,
                max_depth: int = 5,
                min_decrease: float = 0.01,
                max_features: float = 0.8,
                seed: int = None,
                **kwargs) -> np.ndarray:
    """Fit Quantile Random Forest and predict on evaluation grid."""

    domain = np.array([[X_train.min(), X_train.max()]]) if X_train.shape[1] == 1 else \
             np.array([[X_train[:, i].min(), X_train[:, i].max()] for i in range(X_train.shape[1])])

    model = RandomForest(
        n_trees=n_trees,
        domain=domain,
        value_fn=value_quantile,
        max_depth=max_depth,
        min_decrease=min_decrease
    )

    model.max_features = max_features

    model.fit(X_train, y_train, PRNGKey=seed)

    predictions = model.predict(x_grid, q=q_grid, output_shape=len(q_grid))

    return predictions
import sys
import time

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from random import random, randint, choice
from math import floor
import copy

# PG DEFINITIONS AND OPERATORS ---

# Set of Functions and Terminals (For feature synthesis)
FUNCTIONS = ['+', '-', '*', '/_prot', 'sqrt', 'log', 'pow2', 'tanh']
TERMINALS = []  # Will be filled with variables (X_cols) and Constants.
CONST_MIN, CONST_MAX = -5, 5


# Operator mapping to NumPy functions, including protected ones
def prot_div(x1, x2):
    """Protected division to avoid division-by-zero errors."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) < 1e-6, 1.0, x1 / x2)

def protected_sqrt(x):
    return np.sqrt(np.abs(x))

def protected_log(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) < 1e-6, 0.0, np.log(np.abs(x)))

def power_2(x):
    return np.power(x, 2)

def tanh(x):
    return np.tanh(np.nan_to_num(x))

OPERATOR_MAP = {
    '+': np.add,
    '-': np.subtract,
    '*': np.multiply,
    '/_prot': prot_div,
    'sqrt': protected_sqrt,
    'log': protected_log,
    'pow2': power_2,
    'tanh': tanh
}


class Node:
    """Represents a node in the GP expression tree."""

    def __init__(self, value, arity=0):
        self.value = value
        self.arity = arity  # Number of arguments (0 for terminals)
        self.children = []

    def __repr__(self):
        """Simple representation for tree printing."""
        if self.arity == 0:
            return str(self.value)
        if self.arity == 1:
            return f"{self.value}({self.children[0]})"
        else:
            return f"({self.children[0]} {self.value} {self.children[1]})"

    def get_length(self) -> int:
        """
        Calculates and returns the total number of nodes (functions + terminals)
        in the subtree starting at this node.
        """
        # Start counting with the current node (1)
        length = 1

        # Recursively add the length of all children (subtrees)
        for child in self.children:
            length += child.get_length()

        return length


class Individual:
    """Class for a hybrid individual"""

    def __init__(self, mask, trees, num_syn_attr):
        self.mask = mask  # GA Part: Binary mask (Feature Selection)
        self.trees = trees  # GP Part: List of Node objects (Feature Synthesis)
        self.num_syn_attr = num_syn_attr
        self.fitness = -np.inf
        self.error = np.inf
        self.num_total_attr = 0


# GENETIC PROGRAMMING (GP) AUXILIARY FUNCTIONS ---

def create_random_tree(max_depth, functions, terminals, method='grow',
                       current_depth=0):
    """Recursively generates a random expression tree (using 'Grow' method)."""

    # Condition to generate a Terminal (reaching max depth or randomness)
    is_terminal = (current_depth >= max_depth) or (
                random() < 0.2 and current_depth > 0)

    if is_terminal:
        value = choice(terminals)
        if value == 'Cte':
            value = np.round(random() * (CONST_MAX - CONST_MIN) + CONST_MIN, 2)
        node = Node(value, arity=0)
    else:
        # Select Function
        func_symbol = choice(functions)

        if func_symbol in ['sen', 'cos', 'sqrt', 'log', 'pow2', 'tanh']:
            arity = 1
        else:
            arity = 2

        node = Node(func_symbol, arity=arity)

        # Generate children recursively
        for _ in range(arity):
            node.children.append(
                create_random_tree(max_depth, functions, terminals, method,
                                   current_depth + 1))

    return node


def evaluate_tree(tree_node: Node, X_data: np.ndarray):
    """
    Evaluates the expression tree (tree_node) over the data matrix (X_data).
    Returns a one-column array (the new synthesized feature).
    """

    # Terminal Case
    if tree_node.arity == 0:
        if isinstance(tree_node.value, str) and tree_node.value.startswith(
                'X_'):
            # It's a variable (X_i)
            col_index = int(tree_node.value.split('_')[1])
            return X_data[:, col_index]
        else:
            # It's a numerical constant. Creates an array of the same length as the data.
            return np.full(X_data.shape[0], tree_node.value)

    # Function Case
    func = OPERATOR_MAP[tree_node.value]

    # Evaluate children recursively
    evaluated_children = [evaluate_tree(child, X_data) for child in
                          tree_node.children]

    # Apply the NumPy function
    return func(*evaluated_children)


def find_random_node(root: Node):
    """Finds a random node in the tree."""
    nodes = []

    def traverse(node):
        nodes.append(node)
        for child in node.children:
            traverse(child)

    traverse(root)
    if not nodes:
        return None
    return choice(nodes)


def find_parent(root, target):
    """Auxiliary to find the parent of a node in the tree."""
    if root is target:
        return None, None
    for i, child in enumerate(root.children):
        if child is target:
            return root, i
        parent, idx = find_parent(child, target)
        if parent:
            return parent, idx
    return None, None


def crossover_tree(parent1_tree, parent2_tree):
    """Subtree crossover: swaps a random subtree between two parents."""

    # Deep copy of tree 1
    child = copy.deepcopy(parent1_tree)

    # Find random nodes in the child and in parent 2
    node_to_replace = find_random_node(child)
    node_to_insert = find_random_node(parent2_tree)

    if not node_to_replace or not node_to_insert:
        return child

    # Perform the swap

    # Find the parent of the node to replace in the child
    parent_of_replace, index_in_parent = find_parent(child, node_to_replace)

    if parent_of_replace:
        # Insert a deep copy of the node to insert into the position of the node to replace
        parent_of_replace.children[index_in_parent] = copy.deepcopy(
            node_to_insert)
    else:
        # The node to replace was the root
        child = copy.deepcopy(node_to_insert)

    return child


def mutate_tree(tree_node: Node):
    """Subtree mutation: replaces a random subtree with a new random tree."""

    node_to_mutate = find_random_node(tree_node)

    if node_to_mutate:
        # Generate a new random subtree
        new_sub_tree = create_random_tree(3, FUNCTIONS, TERMINALES)

        # Find the parent and replace the node
        parent_of_mutate, index_in_parent = find_parent(tree_node,
                                                        node_to_mutate)

        if parent_of_mutate:
            parent_of_mutate.children[index_in_parent] = new_sub_tree
        else:
            # The node to mutate was the root
            tree_node = new_sub_tree

    return tree_node


# GENETIC ALGORITHM (GA) CONFIGURATION AND FUNCTIONS ---

POPULATION_SIZE = 200
NUM_GENERATIONS = 100
N_BEST_ELITE = POPULATION_SIZE // 5

W_ERROR = 0.9
W_N = 0.05
W_L = 0.05

MAX_TREE_NODES_PENALTY = 100

PROB_CROSS = 0.7
PROB_MUTATE = 0.3
MEAN_SYN_ATTR = 4
STD_SYN_ATTR = 2


def initialize_population(size, n_orig_attrs):
    population = []
    for _ in range(size):
        num_syn_attr = max(0, int(np.round(
            np.random.normal(MEAN_SYN_ATTR, STD_SYN_ATTR))))

        mask = [randint(0, 1) for _ in range(n_orig_attrs)]

        trees = [create_random_tree(5, FUNCTIONS, TERMINALES) for _ in
                 range(num_syn_attr)]

        population.append(Individual(mask, trees, num_syn_attr))
    return population


def evaluate_population(population, X_train, y_train, X_val, y_val,
                        error_base):
    best_fitness = -np.inf
    best_error = np.inf

    for ind in population:
        # GENERATE X_PRIME (X')

        # Selection
        selected_cols = [i for i, bit in enumerate(ind.mask) if bit == 1]
        X_train_selected = X_train[:, selected_cols]
        X_val_selected = X_val[:, selected_cols]

        # Synthesis
        X_train_syn = []
        X_val_syn = []
        for tree in ind.trees:
            # Evaluate the tree to generate the new feature
            try:
                X_train_syn.append(evaluate_tree(tree, X_train).reshape(-1, 1))
                X_val_syn.append(evaluate_tree(tree, X_val).reshape(-1, 1))
            except Exception as e:
                # Handle evaluation errors
                print(f"ERROR evaluating tree: {e}")
                X_train_syn.append(np.zeros((X_train.shape[0], 1)))
                X_val_syn.append(np.zeros((X_val.shape[0], 1)))

        # Combination
        combined_features_train = [X_train_selected] + X_train_syn
        combined_features_val = [X_val_selected] + X_val_syn

        # Handle the case where nothing is selected or synthesized
        if not combined_features_train or all(
                arr.shape[1] == 0 for arr in combined_features_train):
            ind.num_total_attr = 0
            ind.error = np.inf
            ind.fitness = 0.0  # Minimum fitness
            continue

        X_train_prime = np.hstack(combined_features_train)
        X_val_prime = np.hstack(combined_features_val)

        ind.num_total_attr = X_train_prime.shape[1]

        # CALCULATE OPTIMIZED ERROR (e')
        model_prime = LinearRegression().fit(X_train_prime, y_train)
        y_pred_val = model_prime.predict(X_val_prime)
        error_val = mean_absolute_error(y_val, y_pred_val)
        ind.error = error_val

        # CALCULATE FITNESS
        # Error Cost (Minimize relative error)
        error_cost = error_val / error_base if error_base > 0 else error_val

        # Feature Count Cost (Minimize number of features)
        max_possible_features = len(ind.mask) + MEAN_SYN_ATTR
        feature_cost = ind.num_total_attr / max_possible_features

        # Bloating Cost (Minimize total tree size)
        total_tree_length = sum([t.get_length() for t in ind.trees])
        bloating_cost = total_tree_length / MAX_TREE_NODES_PENALTY

        # Total Weighted Cost (Goal: Minimize)
        weighted_cost = (W_ERROR * error_cost) + (W_N * feature_cost) + (
                    W_L * bloating_cost)

        # Fitness (Goal: Maximize)
        ind.fitness = 1 / (1 + weighted_cost)

        if ind.fitness > best_fitness:
            best_fitness = ind.fitness
            best_error = ind.error

    node_lengths = [sum(t.get_length() for t in ind.trees) for ind in population]
    avg_node_length = sum(node_lengths) / len(node_lengths)
    tree_length = [len(ind.trees) for ind in population]
    avg_tree_length = sum(tree_length) / len(tree_length)

    print("avg_node_length: ", avg_node_length)
    print("avg tree length: ", avg_tree_length)
    return population, best_fitness, best_error


def generate_offspring(elite, n_orig_attrs, num_offspring):
    offspring = []

    # Simple tournament selection among the elite (selects 2 parents)
    def select_parent(elite_list):
        # Randomly select 2 from the elite and return the fittest
        idx1, idx2 = np.random.choice(len(elite_list), 2, replace=False)
        return elite_list[idx1] if elite_list[idx1].fitness > elite_list[
            idx2].fitness else elite_list[idx2]

    while len(offspring) < num_offspring:
        p1 = select_parent(elite)
        p2 = select_parent(elite)

        # Clone a parent to use as the base for the child if no crossover occurs
        child = Individual(p1.mask[:], p1.trees[:], p1.num_syn_attr)

        # CROSSOVER
        if random() < PROB_CROSS:
            child = crossover_hybrid(p1, p2, n_orig_attrs)

        # MUTATION
        if random() < PROB_MUTATE:
            mutate_hybrid(child, n_orig_attrs)

        offspring.append(child)

    return offspring


def crossover_hybrid(p1, p2, n_orig_attrs):
    # CROSSOVER 1: Binary Mask (Two-point crossover, GA style)
    cut1, cut2 = sorted(np.random.choice(n_orig_attrs, 2, replace=False))
    new_mask = p1.mask[:cut1] + p2.mask[cut1:cut2] + p1.mask[cut2:]

    # CROSSOVER 2: Number of Synthesized Features (Rounded mean)
    base_num = (p1.num_syn_attr + p2.num_syn_attr) / 2

    # Add random noise
    noise = choice([-2, -1, 0, 1, 2])
    new_num_syn_attr = max(0, int(floor(base_num + noise)))

    # CROSSOVER 3: Trees (Subtree crossover + simple inheritance)
    new_trees = []
    for i in range(new_num_syn_attr):
        if i < p1.num_syn_attr and i < p2.num_syn_attr and random() < 0.7:
            # Apply GP subtree crossover between tree i of p1 and p2
            new_trees.append(crossover_tree(p1.trees[i], p2.trees[i]))
        elif i < p1.num_syn_attr:
            new_trees.append(copy.deepcopy(p1.trees[i]))
        elif i < p2.num_syn_attr:
            new_trees.append(copy.deepcopy(p2.trees[i]))
        else:
            # Create a new random tree if the mean requires more trees
            new_trees.append(create_random_tree(5, FUNCTIONS, TERMINALES))

    return Individual(new_mask, new_trees, new_num_syn_attr)


def mutate_hybrid(individual, n_orig_attrs):
    # MUTATION 1: Binary Mask (Bit flip)
    idx_mask = randint(0, n_orig_attrs - 1)
    individual.mask[idx_mask] = 1 - individual.mask[idx_mask]

    # MUTATION 2: Trees (Subtree mutation, GP style)
    if individual.num_syn_attr > 0:
        idx_tree = randint(0, individual.num_syn_attr - 1)
        # Mutation handles replacing a portion of the tree
        individual.trees[idx_tree] = mutate_tree(individual.trees[idx_tree])


# --- MAIN EXECUTION FUNCTION ---

def run_hybrid_evolutionary_regression(csv_path):
    """Main function to run the hybrid evolutionary algorithm."""
    t0 = time.time()
    # DATA LOADING AND PREPARATION
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("ERROR: CSV file not found.")
        return

    # Assume the last column is the target variable 'y'
    X_orig = data.iloc[:, :-1].values
    y_orig = data.iloc[:, -1].values

    n_original_attrs = X_orig.shape[1]

    # Initialize TERMINALS for GP
    global TERMINALES
    TERMINALES = [f'X_{i}' for i in range(n_original_attrs)] + ['Cte']

    # Simple Train/Validation split
    split_idx = int(0.8 * len(X_orig))
    X_train, X_val = X_orig[:split_idx], X_orig[split_idx:]
    y_train, y_val = y_orig[:split_idx], y_orig[split_idx:]

    print(
        f"Dataset loaded with {n_original_attrs} original features. Size: {len(X_orig)} samples.")

    # CALCULATE BASE ERROR (e)
    if n_original_attrs == 0:
        error_base = np.inf
    else:
        model_base = LinearRegression().fit(X_train, y_train)
        y_pred_base = model_base.predict(X_val)
        error_base = mean_absolute_error(y_val, y_pred_base)

    print(f"Base Error with Original Dataset: {error_base:.4f} (e)")

    # POPULATION INITIALIZATION
    population = initialize_population(POPULATION_SIZE, n_original_attrs)

    # EVOLUTIONARY LOOP
    best_overall_individual = None

    for generation in range(NUM_GENERATIONS):
        print(f"\n--- Generation {generation + 1} ---")

        # POPULATION EVALUATION
        population, best_fitness, best_error = evaluate_population(
            population, X_train, y_train, X_val, y_val, error_base
        )
        print(
            f"Best Gen Error: {best_error:.4f}, Best Gen Fitness: {best_fitness:.4f}")

        # ELITE SELECTION
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        elite = population[:N_BEST_ELITE]

        # Save the global best individual
        if best_overall_individual is None or elite[
            0].fitness > best_overall_individual.fitness:
            best_overall_individual = copy.deepcopy(elite[0])

        # REPRODUCTION (CROSSOVER AND MUTATION)
        offspring = generate_offspring(elite, n_original_attrs,
                                       POPULATION_SIZE - N_BEST_ELITE)

        # REPLACEMENT (Generational Elitist Strategy)
        population = elite + offspring

    # FINAL RESULT
    final_best = best_overall_individual

    t = time.time() - t0

    print("\n--- Final Results ---")
    print(f"Base Error (e): {error_base:.4f}")
    print(f"Optimized Error (e'): {final_best.error:.4f}")
    print(f"Selection Mask: {final_best.mask}")
    print(f"Execution Time: {t}")
    print("Synthesized Formulas (Trees):")
    for tree in final_best.trees:
        print(f"  -> {repr(tree)}")

    if final_best.error < error_base:
        print("\nOPTIMIZATION SUCCESSFUL! e' < e")
    else:
        print("\nOptimization did not improve upon the base error.")

    return final_best

if __name__ == "__main__":
    run_hybrid_evolutionary_regression(sys.argv[1])

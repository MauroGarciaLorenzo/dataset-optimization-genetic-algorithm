import numpy as np


class Node:
    """Representa un nodo en el árbol de expresión PG."""

    def __init__(self, value, arity=0):
        self.value = value
        self.arity = arity  # Número de argumentos (0 para terminales)
        self.children = []

    def __repr__(self):
        """Representación simple para la impresión del árbol."""
        if self.arity == 0:
            return str(self.value)

        # Maneja operadores unarios y binarios
        if self.arity == 1:
            return f"{self.value}({self.children[0]})"
        else:
            return f"({self.children[0]} {self.value} {self.children[1]})"


class Individual:
    """Clase para un individuo híbrido (AG + PG)."""

    def __init__(self, mask, trees, num_syn_attr):
        self.mask = mask  # Parte AG: Máscara binaria (Selección de Features)
        self.trees = trees  # Parte PG: Lista de objetos Node (Síntesis de Features)
        self.num_syn_attr = num_syn_attr
        self.fitness = -np.inf
        self.error = np.inf
        self.num_total_attr = 0
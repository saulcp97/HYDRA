"""
Similarity measures for comparing PyTorch model weight tensors.

This module defines a common interface to compare two tensors and several
implementations that can be used in decentralized federated learning for
dynamic coalition formation.

The measures included are:

- Cosine similarity
- Euclidean distance
- Normalized Euclidean distance
- Manhattan distance
- Pearson correlation
- Angular distance

Notes
-----
Some measures are true similarities, where higher values indicate greater
closeness, while others are distances, where lower values indicate greater
closeness. This is exposed through the ``is_distance`` property.

All measures expect two tensors with the same shape. Internally, tensors are
flattened to 1D before the computation.

The implementation is designed to be compatible with static type checking
(mypy), linting (pylint), and automatic documentation generation with Sphinx.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class SimilarityMeasure(ABC):
    """
    Abstract interface for tensor comparison measures.

    Subclasses must implement :meth:`compute`, which compares two PyTorch
    tensors and returns a scalar result.

    The result may represent either a similarity or a distance depending on
    the concrete implementation. This is indicated by the :attr:`is_distance`
    property.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the measure.

        Returns
        -------
        str
            Human-readable measure name.
        """

    @property
    @abstractmethod
    def is_distance(self) -> bool:
        """
        Indicate whether the measure is a distance.

        Returns
        -------
        bool
            ``True`` if lower values imply more similarity, ``False`` if
            higher values imply more similarity.
        """

    @abstractmethod
    def compute(self, tensor_a: Tensor, tensor_b: Tensor) -> float:
        """
        Compute the comparison score between two tensors.

        Parameters
        ----------
        tensor_a : torch.Tensor
            First tensor to compare.
        tensor_b : torch.Tensor
            Second tensor to compare.

        Returns
        -------
        float
            Scalar score representing either similarity or distance,
            depending on the concrete implementation.

        Raises
        ------
        ValueError
            If the tensors do not have the same shape or are empty.
        """

    def _validate_and_flatten(self, tensor_a: Tensor, tensor_b: Tensor) -> tuple[Tensor, Tensor]:
        """
        Validate input tensors and flatten them to one dimension.

        Parameters
        ----------
        tensor_a : torch.Tensor
            First tensor to compare.
        tensor_b : torch.Tensor
            Second tensor to compare.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple with both tensors flattened and converted to ``torch.float64``.

        Raises
        ------
        ValueError
            If the tensors do not have the same shape or are empty.
        """
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"Tensors must have the same shape, got {tensor_a.shape} and {tensor_b.shape}."
            )

        if tensor_a.numel() == 0:
            raise ValueError("Tensors must not be empty.")

        flat_a: Tensor = tensor_a.reshape(-1).to(dtype=torch.float64)
        flat_b: Tensor = tensor_b.reshape(-1).to(dtype=torch.float64)
        return flat_a, flat_b


class CosineSimilarityMeasure(SimilarityMeasure):
    """
    Cosine similarity between two tensors.

    This measure evaluates the angular alignment between the flattened tensors:

    .. math::

        \\cos(x, y) = \\frac{x^T y}{\\|x\\|_2 \\|y\\|_2}

    Higher values indicate greater similarity. The output is typically in
    the range ``[-1, 1]``.

    Notes
    -----
    If one or both tensors have zero norm, the method returns ``0.0``.
    """

    @property
    def name(self) -> str:
        """
        Return the name of the measure.

        Returns
        -------
        str
            Measure name.
        """
        return "cosine_similarity"

    @property
    def is_distance(self) -> bool:
        """
        Indicate whether the measure is a distance.

        Returns
        -------
        bool
            Always ``False`` for cosine similarity.
        """
        return False

    def compute(self, tensor_a: Tensor, tensor_b: Tensor) -> float:
        """
        Compute cosine similarity between two tensors.

        Parameters
        ----------
        tensor_a : torch.Tensor
            First tensor to compare.
        tensor_b : torch.Tensor
            Second tensor to compare.

        Returns
        -------
        float
            Cosine similarity score. Higher values indicate greater similarity.
        """
        flat_a, flat_b = self._validate_and_flatten(tensor_a, tensor_b)

        norm_a: Tensor = torch.linalg.norm(flat_a, ord=2)
        norm_b: Tensor = torch.linalg.norm(flat_b, ord=2)

        if torch.isclose(norm_a, torch.tensor(0.0, dtype=norm_a.dtype)) or torch.isclose(
            norm_b, torch.tensor(0.0, dtype=norm_b.dtype)
        ):
            return 0.0

        score: Tensor = torch.dot(flat_a, flat_b) / (norm_a * norm_b)
        return float(score.item())


class EuclideanDistanceMeasure(SimilarityMeasure):
    """
    Euclidean distance between two tensors.

    This measure computes the L2 norm of the difference between the flattened
    tensors:

    .. math::

        d(x, y) = \\|x - y\\|_2

    Lower values indicate greater similarity.
    """

    @property
    def name(self) -> str:
        """
        Return the name of the measure.

        Returns
        -------
        str
            Measure name.
        """
        return "euclidean_distance"

    @property
    def is_distance(self) -> bool:
        """
        Indicate whether the measure is a distance.

        Returns
        -------
        bool
            Always ``True`` for Euclidean distance.
        """
        return True

    def compute(self, tensor_a: Tensor, tensor_b: Tensor) -> float:
        """
        Compute Euclidean distance between two tensors.

        Parameters
        ----------
        tensor_a : torch.Tensor
            First tensor to compare.
        tensor_b : torch.Tensor
            Second tensor to compare.

        Returns
        -------
        float
            Euclidean distance. Lower values indicate greater similarity.
        """
        flat_a, flat_b = self._validate_and_flatten(tensor_a, tensor_b)
        distance: Tensor = torch.linalg.norm(flat_a - flat_b, ord=2)
        return float(distance.item())
    
class NormalizedEuclideanDistanceMeasure(SimilarityMeasure):
    """
    Normalized Euclidean distance between two tensors.

    This measure normalizes the Euclidean distance by the sum of the L2 norms
    of the tensors:

    .. math::

        d_{norm}(x, y) = \\frac{\\|x - y\\|_2}{\\|x\\|_2 + \\|y\\|_2}

    Lower values indicate greater similarity.

    Notes
    -----
    If both tensors are zero vectors, the method returns ``0.0``.
    """

    @property
    def name(self) -> str:
        """
        Return the name of the measure.

        Returns
        -------
        str
            Measure name.
        """
        return "normalized_euclidean_distance"

    @property
    def is_distance(self) -> bool:
        """
        Indicate whether the measure is a distance.

        Returns
        -------
        bool
            Always ``True`` for normalized Euclidean distance.
        """
        return True

    def compute(self, tensor_a: Tensor, tensor_b: Tensor) -> float:
        """
        Compute normalized Euclidean distance between two tensors.

        Parameters
        ----------
        tensor_a : torch.Tensor
            First tensor to compare.
        tensor_b : torch.Tensor
            Second tensor to compare.

        Returns
        -------
        float
            Normalized Euclidean distance. Lower values indicate greater
            similarity.
        """
        flat_a, flat_b = self._validate_and_flatten(tensor_a, tensor_b)

        numerator: Tensor = torch.linalg.norm(flat_a - flat_b, ord=2)
        denominator: Tensor = torch.linalg.norm(flat_a, ord=2) + torch.linalg.norm(flat_b, ord=2)

        if torch.isclose(denominator, torch.tensor(0.0, dtype=denominator.dtype)):
            return 0.0

        score: Tensor = numerator / denominator
        return float(score.item())


class ManhattanDistanceMeasure(SimilarityMeasure):
    """
    Manhattan distance between two tensors.

    This measure computes the L1 norm of the difference between the flattened
    tensors:

    .. math::

        d(x, y) = \\|x - y\\|_1

    Lower values indicate greater similarity.
    """

    @property
    def name(self) -> str:
        """
        Return the name of the measure.

        Returns
        -------
        str
            Measure name.
        """
        return "manhattan_distance"

    @property
    def is_distance(self) -> bool:
        """
        Indicate whether the measure is a distance.

        Returns
        -------
        bool
            Always ``True`` for Manhattan distance.
        """
        return True

    def compute(self, tensor_a: Tensor, tensor_b: Tensor) -> float:
        """
        Compute Manhattan distance between two tensors.

        Parameters
        ----------
        tensor_a : torch.Tensor
            First tensor to compare.
        tensor_b : torch.Tensor
            Second tensor to compare.

        Returns
        -------
        float
            Manhattan distance. Lower values indicate greater similarity.
        """
        flat_a, flat_b = self._validate_and_flatten(tensor_a, tensor_b)
        distance: Tensor = torch.linalg.norm(flat_a - flat_b, ord=1)
        return float(distance.item())

class PearsonCorrelationMeasure(SimilarityMeasure):
    """
    Pearson correlation coefficient between two tensors.

    This measure computes the correlation between centered flattened tensors:

    .. math::

        \\rho(x, y) = \\frac{(x - \\bar{x})^T (y - \\bar{y})}
        {\\|x - \\bar{x}\\|_2 \\|y - \\bar{y}\\|_2}

    Higher values indicate greater similarity. The output is typically in the
    range ``[-1, 1]``.

    Notes
    -----
    If one or both centered tensors have zero norm, the method returns ``0.0``.
    """

    @property
    def name(self) -> str:
        """
        Return the name of the measure.

        Returns
        -------
        str
            Measure name.
        """
        return "pearson_correlation"

    @property
    def is_distance(self) -> bool:
        """
        Indicate whether the measure is a distance.

        Returns
        -------
        bool
            Always ``False`` for Pearson correlation.
        """
        return False

    def compute(self, tensor_a: Tensor, tensor_b: Tensor) -> float:
        """
        Compute Pearson correlation between two tensors.

        Parameters
        ----------
        tensor_a : torch.Tensor
            First tensor to compare.
        tensor_b : torch.Tensor
            Second tensor to compare.

        Returns
        -------
        float
            Pearson correlation coefficient. Higher values indicate greater
            similarity.
        """
        flat_a, flat_b = self._validate_and_flatten(tensor_a, tensor_b)

        centered_a: Tensor = flat_a - torch.mean(flat_a)
        centered_b: Tensor = flat_b - torch.mean(flat_b)

        norm_a: Tensor = torch.linalg.norm(centered_a, ord=2)
        norm_b: Tensor = torch.linalg.norm(centered_b, ord=2)

        if torch.isclose(norm_a, torch.tensor(0.0, dtype=norm_a.dtype)) or torch.isclose(
            norm_b, torch.tensor(0.0, dtype=norm_b.dtype)
        ):
            return 0.0

        score: Tensor = torch.dot(centered_a, centered_b) / (norm_a * norm_b)
        return float(score.item())

class AngularDistanceMeasure(SimilarityMeasure):
    """
    Angular distance between two tensors.

    This measure derives a distance from cosine similarity:

    .. math::

        d_{ang}(x, y) = \\frac{\\arccos(\\cos(x, y))}{\\pi}

    Lower values indicate greater similarity. The output is in the range
    ``[0, 1]``.

    Notes
    -----
    If one or both tensors have zero norm, the method returns ``0.5`` as a
    neutral default value.
    """

    def __init__(self) -> None:
        """
        Initialize the angular distance measure.
        """
        self._cosine = CosineSimilarityMeasure()

    @property
    def name(self) -> str:
        """
        Return the name of the measure.

        Returns
        -------
        str
            Measure name.
        """
        return "angular_distance"

    @property
    def is_distance(self) -> bool:
        """
        Indicate whether the measure is a distance.

        Returns
        -------
        bool
            Always ``True`` for angular distance.
        """
        return True

    def compute(self, tensor_a: Tensor, tensor_b: Tensor) -> float:
        """
        Compute angular distance between two tensors.

        Parameters
        ----------
        tensor_a : torch.Tensor
            First tensor to compare.
        tensor_b : torch.Tensor
            Second tensor to compare.

        Returns
        -------
        float
            Angular distance in the range ``[0, 1]``. Lower values indicate
            greater similarity.
        """
        flat_a, flat_b = self._validate_and_flatten(tensor_a, tensor_b)

        norm_a: Tensor = torch.linalg.norm(flat_a, ord=2)
        norm_b: Tensor = torch.linalg.norm(flat_b, ord=2)

        if torch.isclose(norm_a, torch.tensor(0.0, dtype=norm_a.dtype)) or torch.isclose(
            norm_b, torch.tensor(0.0, dtype=norm_b.dtype)
        ):
            return 0.5

        cosine_value: float = self._cosine.compute(flat_a, flat_b)
        clamped: float = max(-1.0, min(1.0, cosine_value))
        angle: float = torch.arccos(torch.tensor(clamped, dtype=torch.float64)).item()
        return float(angle / torch.pi)


class SimilarityMeasureFactory:
    """
    Factory utility to build similarity measure instances by name.
    """

    _REGISTRY: dict[str, type[SimilarityMeasure]] = {
        "cosine_similarity": CosineSimilarityMeasure,
        "euclidean_distance": EuclideanDistanceMeasure,
        "normalized_euclidean_distance": NormalizedEuclideanDistanceMeasure,
        "manhattan_distance": ManhattanDistanceMeasure,
        "pearson_correlation": PearsonCorrelationMeasure,
        "angular_distance": AngularDistanceMeasure,
    }

    @classmethod
    def create(cls, name: str) -> SimilarityMeasure:
        """
        Create a similarity measure instance from its name.

        Parameters
        ----------
        name : str
            Name of the measure.

        Returns
        -------
        SimilarityMeasure
            Instantiated similarity measure.

        Raises
        ------
        ValueError
            If the requested measure name is not registered.
        """
        try:
            measure_class = cls._REGISTRY[name]
        except KeyError as exc:
            available = ", ".join(sorted(cls._REGISTRY.keys()))
            raise ValueError(
                f"Unknown similarity measure '{name}'. Available measures: {available}."
            ) from exc

        return measure_class()

"""
if __name__ == "__main__":
    tensor_1 = torch.tensor([1.0, 2.0, 3.0])
    tensor_2 = torch.tensor([1.1, 1.9, 3.2])

    measures: list[SimilarityMeasure] = [
        CosineSimilarityMeasure(),
        EuclideanDistanceMeasure(),
        NormalizedEuclideanDistanceMeasure(),
        ManhattanDistanceMeasure(),
        PearsonCorrelationMeasure(),
        AngularDistanceMeasure(),
    ]

    for measure in measures:
        result = measure.compute(tensor_1, tensor_2)
        kind = "distance" if measure.is_distance else "similarity"
        print(f"{measure.name} ({kind}): {result:.6f}")
"""
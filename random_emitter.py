"""Provides the RandomEmitter."""
import numpy as np
from ribs.emitters._emitter_base import EmitterBase

class RandomEmitter(EmitterBase):
    """Emits solutions by sampling uniformly at random.
    If the archive is empty, calls to :meth:`ask` will return the initial `x0`.
    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (array-like): Initial point.
        bounds (array-like): Bounds of the solution space.
            Array-like to specify the bounds for each
            dim. Each element in this array-like should be a tuple of
            ``(lower_bound, upper_bound)``.
        batch_size (int): Number of solutions to return in :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: There is an error in the bounds configuration.
    """

    def __init__(self,
                 archive,
                 x0,
                 bounds=None,
                 batch_size=64,
                 seed=None):
        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size
        self._x0 = np.array(x0, dtype=archive.dtype)

        EmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
        )

    @property
    def x0(self):
        """numpy.ndarray: Initial starting point."""
        return self._x0

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    def ask(self):
        """Creates solutions by sampling uniformly at random.
        Returns:
            ``(batch_size, solution_dim)`` array -- contains ``batch_size`` new
            solutions to evaluate.
        """
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self._batch_size, len(self._x0)))


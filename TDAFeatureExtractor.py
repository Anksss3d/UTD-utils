import numpy as np
import cv2
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve


class TDAFeatureExtractor:
    """
    A simple class for TDA-based feature extraction using giotto-tda.
    This version supports:
      - Single-channel (grayscale) extraction
      - Multi-channel (RGB) extraction, channel by channel
    """

    def __init__(self, homology_dims=(0, 1), coeff=3, n_bins=100, n_jobs=-1):
        """
        :param homology_dims: Tuple of homology dimensions to compute (default is (0,1))
        :param coeff: Coefficient for persistent homology (default=3)
        :param n_bins: Number of bins used in BettiCurve (default=100)
        :param n_jobs: Number of parallel jobs (default=-1 -> use all available cores)
        """
        self.homology_dims = homology_dims
        self.coeff = coeff
        self.n_bins = n_bins
        self.n_jobs = n_jobs

        # Instantiate these once for efficiency
        self.cubical_persistence = CubicalPersistence(
            homology_dimensions=self.homology_dims,
            coeff=self.coeff,
            n_jobs=self.n_jobs
        )
        self.betti_curve = BettiCurve(n_bins=self.n_bins)

    def generate_betti_input(self, img: np.ndarray, multi_channel: bool = False) -> np.ndarray:
        """
        Generate Betti curve features from the given image.

        If multi_channel=False (default), the method converts the image to
        grayscale (if not already) and computes Betti curves for the single channel.

        If multi_channel=True, the method splits the image by channel and computes
        Betti curves per channel, returning a stacked feature vector.

        :param img: A NumPy array with shape (H, W) or (H, W, C)
        :param multi_channel: Whether to compute features per channel (True)
                              or just on grayscale (False).
        :return: A NumPy array of Betti curve features.
                 - single channel shape typically (1, n_bins)
                 - multi-channel shape typically (1, n_bins * C)
        """

        # If the image is already single-channel, shape is (H, W).
        # If it's multi-channel, shape is (H, W, C).
        if len(img.shape) == 2:
            # Already single-channel
            return self._betti_single_channel(img)

        # If here, shape is at least (H, W, 3) or something similar
        if not multi_channel:
            # Convert to grayscale for single-channel
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return self._betti_single_channel(img_gray)
        else:
            # Process each channel separately, then stack the results
            # For example, if the image is (H, W, 3), we'll get 3 sets of Betti curves
            channels = cv2.split(img)  # returns a tuple of channel arrays
            all_channel_features = []
            for ch in channels:
                # ch shape is (H, W)
                betti_ch = self._betti_single_channel(ch)
                all_channel_features.append(betti_ch)

            # betti_ch typically has shape (1, n_bins)
            # We can concatenate them horizontally to get (1, n_bins*C)
            return np.hstack(all_channel_features)

    def _betti_single_channel(self, single_channel_img: np.ndarray) -> np.ndarray:
        """
        Internal helper: run CubicalPersistence + BettiCurve on a single-channel image.
        Returns a 2D array of shape (1, n_bins).
        """
        # Ensure it has shape (1, H, W) for giotto-tda
        single_channel_img = single_channel_img[None, :, :]

        diagrams = self.cubical_persistence.fit_transform(single_channel_img)
        betti_curves = self.betti_curve.fit_transform(diagrams)

        return betti_curves  # shape typically (1, n_bins)

    def calculate_betti_numbers(self, img: np.ndarray, thresholds: range) -> (list, list):
        """
        Calculate Betti-0 and Betti-1 numbers by thresholding the image at each value in 'thresholds'.
        (Currently uses grayscale thresholding, ignoring multi-channel.)

        :param img: Input image (H, W) or (H, W, 3)
        :param thresholds: A range or list of thresholds (e.g., range(1, 6))
        :return: Two lists: betti_0_numbers, betti_1_numbers
        """
        # If multi-channel, convert to grayscale before thresholding
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        betti_0_numbers = []
        betti_1_numbers = []

        for threshold in thresholds:
            # Binarize based on the threshold
            binary_image = (img_gray >= threshold).astype(np.int32)

            # Create a fresh CubicalPersistence for each threshold
            cubical_persistence = CubicalPersistence(
                homology_dimensions=[0, 1],
                coeff=self.coeff,
                n_jobs=self.n_jobs
            )

            diagrams = cubical_persistence.fit_transform(binary_image[None, :, :])
            # diagrams[0] shape: (n_points, 3) => (birth, death, dimension)

            # Betti-0: # of points where dimension == 0
            betti_0 = sum(1 for point in diagrams[0] if point[2] == 0)
            # Betti-1: # of points where dimension == 1
            betti_1 = sum(1 for point in diagrams[0] if point[2] == 1)

            betti_0_numbers.append(betti_0)
            betti_1_numbers.append(betti_1)

        return betti_0_numbers, betti_1_numbers

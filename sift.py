"""
SIFT Feature Extraction Implementation from Scratch
Author: Computer Vision Course
Description: Complete implementation of Scale-Invariant Feature Transform (SIFT)
             including scale-space construction, keypoint detection, and descriptor computation.
             Includes RANSAC for robust homography estimation.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import warnings


class SIFT:
    """
    Scale-Invariant Feature Transform (SIFT) implementation.
    
    This class implements the SIFT algorithm for detecting and describing
    distinctive keypoints in images that are invariant to scale, rotation,
    and changes in illumination.
    """
    
    def __init__(
        self,
        n_octaves: int = 4,
        n_scales: int = 3,
        sigma: float = 1.6,
        contrast_threshold: float = 0.02,  # Lower = more keypoints (was 0.04)
        edge_threshold: float = 10.0,
    ):
        """
        Initialize SIFT detector.
        
        Args:
            n_octaves: Number of octaves in the scale-space pyramid
            n_scales: Number of scale levels per octave (S in SIFT paper)
            sigma: Initial Gaussian blur standard deviation
            contrast_threshold: Threshold for keypoint contrast (lower = more keypoints)
            edge_threshold: Threshold for edge response (to filter edge-like features)
        """
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.sigma = sigma
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        
        # Pre-compute Gaussian blur parameters
        # We need n_scales + 3 blurred images per octave
        self.k = 2 ** (1.0 / n_scales)  # Spacing between scale levels
        self.sigmas = [sigma * (self.k ** i) for i in range(n_scales + 3)]
    
    def _build_gaussian_pyramid(self, image: np.ndarray) -> List[List[np.ndarray]]:
        """
        Build Gaussian pyramid.
        
        For each octave, we create n_scales+3 Gaussian blurred images
        at progressively larger blur levels.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            List of octaves, each containing list of blurred images
        """
        pyramid = []
        
        # Convert to float and normalize
        base = image.astype(np.float32) / 255.0
        
        # Apply initial blur to compensate for assumed camera blur (sigma = 0.5)
        # Target sigma for first level is self.sigma
        sigma_diff = np.sqrt(max(self.sigma ** 2 - 0.5 ** 2, 0.01))
        current = cv2.GaussianBlur(base, (0, 0), sigma_diff)
        
        for octave_idx in range(self.n_octaves):
            octave = [current]  # First image in octave
            
            # Build remaining scales in this octave
            for scale_idx in range(1, len(self.sigmas)):
                # Compute incremental sigma
                prev_sigma = self.sigmas[scale_idx - 1]
                curr_sigma = self.sigmas[scale_idx]
                sigma_diff = np.sqrt(curr_sigma ** 2 - prev_sigma ** 2)
                
                # Apply incremental blur to previous image
                blurred = cv2.GaussianBlur(octave[-1], (0, 0), sigma_diff)
                octave.append(blurred)
            
            pyramid.append(octave)
            
            # Downsample for next octave (use image from middle of current octave)
            if octave_idx < self.n_octaves - 1:
                current = octave[self.n_scales][::2, ::2]
        
        return pyramid
    
    def _build_dog_pyramid(self, gaussian_pyramid: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        Build Difference of Gaussians (DoG) pyramid.
        
        DoG is computed as the difference between consecutive Gaussian blurred images.
        
        Args:
            gaussian_pyramid: Gaussian pyramid from _build_gaussian_pyramid
            
        Returns:
            List of octaves, each containing DoG images
        """
        dog_pyramid = []
        
        for octave in gaussian_pyramid:
            dog_octave = []
            for i in range(len(octave) - 1):
                dog = octave[i + 1] - octave[i]
                dog_octave.append(dog)
            dog_pyramid.append(dog_octave)
        
        return dog_pyramid
    
    def _find_keypoint_candidates(
        self, dog_pyramid: List[List[np.ndarray]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find keypoint candidates by locating extrema in the DoG pyramid.
        
        A pixel is an extremum if it's a local maximum or minimum in a 3x3x3
        neighborhood (3x3 spatial, 3 scale levels).
        
        Args:
            dog_pyramid: DoG pyramid from _build_dog_pyramid
            
        Returns:
            List of (octave, scale, y, x) tuples for candidate keypoints
        """
        keypoints = []
        
        for octave_idx, octave in enumerate(dog_pyramid):
            # Process middle scales (skip first and last)
            for scale_idx in range(1, len(octave) - 1):
                current = octave[scale_idx]
                prev = octave[scale_idx - 1]
                next_scale = octave[scale_idx + 1]
                
                h, w = current.shape
                
                # Check each pixel (excluding borders)
                for y in range(1, h - 1):
                    for x in range(1, w - 1):
                        pixel_value = current[y, x]
                        
                        # Extract 3x3x3 neighborhood
                        neighborhood = np.concatenate([
                            prev[y-1:y+2, x-1:x+2].flatten(),
                            current[y-1:y+2, x-1:x+2].flatten(),
                            next_scale[y-1:y+2, x-1:x+2].flatten(),
                        ])
                        
                        # Remove center pixel for comparison
                        neighborhood = np.concatenate([
                            neighborhood[:13],
                            neighborhood[14:]
                        ])
                        
                        # Check if extremum
                        if pixel_value > 0:
                            if pixel_value > np.max(neighborhood):
                                keypoints.append((octave_idx, scale_idx, y, x))
                        else:
                            if pixel_value < np.min(neighborhood):
                                keypoints.append((octave_idx, scale_idx, y, x))
        
        return keypoints
    
    def _refine_keypoint_location(
        self,
        dog_pyramid: List[List[np.ndarray]],
        octave_idx: int,
        scale_idx: int,
        y: int,
        x: int,
    ) -> Optional[Tuple[int, int, int, float, float]]:
        """
        Refine keypoint location using Taylor expansion.
        
        Fits a quadratic function to the local DoG values and finds the sub-pixel
        location of the extremum. Also filters out low-contrast keypoints.
        
        Args:
            dog_pyramid: DoG pyramid
            octave_idx: Octave index
            scale_idx: Scale index within octave
            y, x: Integer coordinates of candidate
            
        Returns:
            (y, x, scale_idx, scale, contrast) or None if rejected
        """
        octave = dog_pyramid[octave_idx]
        offset = np.array([0.0, 0.0, 0.0])
        
        # Refine position using quadratic fit
        for refinement_iter in range(5):
            # Check bounds first
            if (scale_idx < 1 or scale_idx >= len(octave) - 1):
                return None
            
            h, w = octave[scale_idx].shape
            if (y < 1 or y >= h - 1 or x < 1 or x >= w - 1):
                return None
            
            current = octave[scale_idx]
            prev = octave[scale_idx - 1]
            next_scale = octave[scale_idx + 1]
            
            # Compute gradient vector
            gx = (current[y, x + 1] - current[y, x - 1]) / 2.0
            gy = (current[y + 1, x] - current[y - 1, x]) / 2.0
            gs = (next_scale[y, x] - prev[y, x]) / 2.0
            
            gradient = np.array([gx, gy, gs])
            
            # Compute Hessian matrix
            gxx = current[y, x + 1] - 2 * current[y, x] + current[y, x - 1]
            gyy = current[y + 1, x] - 2 * current[y, x] + current[y - 1, x]
            gss = next_scale[y, x] - 2 * current[y, x] + prev[y, x]
            
            gxy = (current[y + 1, x + 1] - current[y + 1, x - 1] -
                   current[y - 1, x + 1] + current[y - 1, x - 1]) / 4.0
            gxs = (next_scale[y, x + 1] - next_scale[y, x - 1] -
                   prev[y, x + 1] + prev[y, x - 1]) / 4.0
            gys = (next_scale[y + 1, x] - next_scale[y - 1, x] -
                   prev[y + 1, x] + prev[y - 1, x]) / 4.0
            
            hessian = np.array([
                [gxx, gxy, gxs],
                [gxy, gyy, gys],
                [gxs, gys, gss]
            ])
            
            # Solve for offset: H * offset = -g
            try:
                offset = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                return None
            
            # Check if offset is small (converged)
            if np.abs(offset).max() < 0.5:
                break
            
            # Update position
            x += int(np.round(offset[0]))
            y += int(np.round(offset[1]))
            scale_idx += int(np.round(offset[2]))
        
        # Check bounds after final update
        if (scale_idx < 1 or scale_idx >= len(octave) - 1):
            return None
            
        h, w = octave[scale_idx].shape
        if (y < 1 or y >= h - 1 or x < 1 or x >= w - 1):
            return None
        
        # Compute contrast
        current = octave[scale_idx]
        contrast = current[y, x] + 0.5 * np.dot(gradient, offset)
        
        # Reject low-contrast keypoints
        if np.abs(contrast) < self.contrast_threshold:
            return None
        
        scale = self.sigma * (self.k ** scale_idx) * (2 ** octave_idx)
        
        return y, x, scale_idx, scale, contrast
    
    def _is_edge_like(
        self,
        dog_pyramid: List[List[np.ndarray]],
        octave_idx: int,
        scale_idx: int,
        y: int,
        x: int,
    ) -> bool:
        """
        Reject keypoints that lie on edges.
        
        Uses the ratio of the principal curvatures via the Hessian matrix.
        Edges have one large and one small principal curvature.
        
        Args:
            dog_pyramid: DoG pyramid
            octave_idx, scale_idx, y, x: Keypoint coordinates
            
        Returns:
            True if the keypoint is edge-like (should be rejected)
        """
        current = dog_pyramid[octave_idx][scale_idx]
        h, w = current.shape
        
        # Ensure we're within bounds
        if y < 1 or y >= h - 1 or x < 1 or x >= w - 1:
            return True
        
        # Hessian (2D, spatial only)
        gxx = current[y, x + 1] - 2 * current[y, x] + current[y, x - 1]
        gyy = current[y + 1, x] - 2 * current[y, x] + current[y - 1, x]
        gxy = (current[y + 1, x + 1] - current[y + 1, x - 1] -
               current[y - 1, x + 1] + current[y - 1, x - 1]) / 4.0
        
        trace = gxx + gyy
        det = gxx * gyy - gxy ** 2
        
        # Avoid division by zero
        if det <= 0:
            return True
        
        # Ratio of principal curvatures
        ratio = (trace ** 2) / det
        threshold = ((self.edge_threshold + 1) ** 2) / self.edge_threshold
        
        return ratio > threshold
    
    def _compute_orientation(
        self,
        gaussian_pyramid: List[List[np.ndarray]],
        octave_idx: int,
        scale_idx: int,
        y: int,
        x: int,
        scale: float,
    ) -> List[float]:
        """
        Compute dominant orientation(s) of the keypoint.
        
        Creates a histogram of gradient orientations weighted by magnitude
        in a circular region around the keypoint.
        
        Args:
            gaussian_pyramid: Gaussian pyramid
            octave_idx, scale_idx, y, x: Keypoint coordinates
            scale: Scale of the keypoint
            
        Returns:
            List of dominant orientations in degrees
        """
        image = gaussian_pyramid[octave_idx][scale_idx]
        h, w = image.shape
        
        # Compute gradients
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = np.arctan2(gy, gx)
        
        # Histogram parameters
        radius = int(3 * scale)
        hist_bins = 36
        hist = np.zeros(hist_bins)
        
        # Weighted histogram of orientations
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                py = int(y + dy)
                px = int(x + dx)
                
                if 0 <= py < h and 0 <= px < w:
                    # Gaussian weight
                    weight = np.exp(-(dx ** 2 + dy ** 2) / (2 * (scale ** 2)))
                    
                    # Add to histogram
                    angle = np.degrees(orientation[py, px]) % 360
                    bin_idx = int(angle / (360 / hist_bins)) % hist_bins
                    hist[bin_idx] += weight * magnitude[py, px]
        
        # Find dominant orientations (peaks in histogram)
        orientations = []
        threshold = 0.8 * np.max(hist)
        
        for i in range(hist_bins):
            if hist[i] > threshold:
                # Fit parabola to get sub-bin accuracy
                left = hist[(i - 1) % hist_bins]
                center = hist[i]
                right = hist[(i + 1) % hist_bins]
                
                angle = (i + 0.5 * (left - right) / (left - 2 * center + right)) * (360 / hist_bins)
                angle = angle % 360
                orientations.append(angle)
        
        return orientations if orientations else [0.0]
    
    def _compute_descriptor(
        self,
        gaussian_pyramid: List[List[np.ndarray]],
        octave_idx: int,
        scale_idx: int,
        y: int,
        x: int,
        orientation: float,
        scale: float,
    ) -> np.ndarray:
        """
        Compute SIFT descriptor for a keypoint.
        
        Creates a 4x4 grid of histogram bins (each with 8 orientation bins),
        resulting in a 128-dimensional descriptor vector.
        
        Args:
            gaussian_pyramid: Gaussian pyramid
            octave_idx, scale_idx, y, x: Keypoint coordinates
            orientation: Dominant orientation in degrees
            scale: Scale of the keypoint
            
        Returns:
            128-dimensional descriptor vector (normalized)
        """
        image = gaussian_pyramid[octave_idx][scale_idx]
        h, w = image.shape
        
        # Compute gradients
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx)
        
        # Descriptor parameters (4x4 grid, 8 orientation bins each = 128 dimensions)
        hist_bins = 8
        descriptor = np.zeros((4, 4, hist_bins))
        
        # Rotation angle
        cos_o = np.cos(np.radians(-orientation))
        sin_o = np.sin(np.radians(-orientation))
        
        # Window size: 16x16 pixels scaled by keypoint scale
        half_width = 8
        
        # Sample in a 16x16 window around the keypoint
        for i in range(-half_width, half_width):
            for j in range(-half_width, half_width):
                # Rotate relative coordinates
                rot_i = cos_o * i - sin_o * j
                rot_j = sin_o * i + cos_o * j
                
                # Map to image coordinates
                py = int(y + rot_i)
                px = int(x + rot_j)
                
                if 0 <= py < h and 0 <= px < w:
                    # Compute rotated gradient angle
                    grad_angle = (angle[py, px] - np.radians(orientation)) % (2 * np.pi)
                    
                    # Bin index for orientation (0-7)
                    bin_idx = int(8 * grad_angle / (2 * np.pi)) % hist_bins
                    
                    # Position in descriptor grid (divide window into 4x4 subregions)
                    grid_i = int((i + half_width) / 4)
                    grid_j = int((j + half_width) / 4)
                    
                    # Clamp to valid range
                    grid_i = np.clip(grid_i, 0, 3)
                    grid_j = np.clip(grid_j, 0, 3)
                    
                    # Gaussian weighting (sigma = half window width)
                    sigma = half_width / 2
                    weight = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                    
                    # Add to histogram
                    descriptor[grid_i, grid_j, bin_idx] += weight * magnitude[py, px]
        
        # Flatten and normalize
        descriptor = descriptor.flatten()
        
        # Normalize to unit length
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        
        # Clip values to 0.2 and re-normalize (reduces effects of illumination changes)
        descriptor = np.minimum(descriptor, 0.2)
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        
        # Convert to uint8 format (scale to 0-255)
        descriptor = (descriptor * 255).astype(np.uint8)
        
        return descriptor
    
    def detectAndCompute(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[List, np.ndarray]:
        """
        Detect keypoints and compute their descriptors.
        
        Args:
            image: Input image (BGR from OpenCV, will be converted to grayscale)
            mask: Optional mask to restrict detection region
            
        Returns:
            (keypoints, descriptors) where:
                - keypoints: List of cv2.KeyPoint objects
                - descriptors: Nx128 array of uint8 descriptors
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Build pyramids
        gaussian_pyramid = self._build_gaussian_pyramid(gray)
        dog_pyramid = self._build_dog_pyramid(gaussian_pyramid)
        
        # Find keypoint candidates
        candidates = self._find_keypoint_candidates(dog_pyramid)
        
        keypoints = []
        descriptors = []
        
        for octave_idx, scale_idx, y, x in candidates:
            # Refine location
            refined = self._refine_keypoint_location(
                dog_pyramid, octave_idx, scale_idx, y, x
            )
            
            if refined is None:
                continue
            
            y_refined, x_refined, scale_idx_refined, scale, contrast = refined
            
            # Reject edge-like keypoints
            if self._is_edge_like(dog_pyramid, octave_idx, scale_idx_refined, y_refined, x_refined):
                continue
            
            # Compute orientation(s)
            orientations = self._compute_orientation(
                gaussian_pyramid, octave_idx, scale_idx_refined, y_refined, x_refined, scale
            )
            
            # Create keypoint for each orientation
            for orientation in orientations:
                # Compute descriptor
                descriptor = self._compute_descriptor(
                    gaussian_pyramid,
                    octave_idx,
                    scale_idx_refined,
                    y_refined,
                    x_refined,
                    orientation,
                    scale,
                )
                
                # Scale coordinates back to original image
                kpt_x = x_refined * (2 ** octave_idx)
                kpt_y = y_refined * (2 ** octave_idx)
                
                kp = cv2.KeyPoint(float(kpt_x), float(kpt_y), float(scale), float(orientation))
                keypoints.append(kp)
                descriptors.append(descriptor)
        
        descriptors = np.array(descriptors) if descriptors else np.empty((0, 128), dtype=np.uint8)
        
        return keypoints, descriptors


class RANSAC:
    """
    Random Sample Consensus (RANSAC) algorithm for robust homography estimation.
    
    RANSAC is used to find the best homography that maps points from one image
    to corresponding points in another image, while being robust to outliers.
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        threshold: float = 3.0,
        confidence: float = 0.99,
    ):
        """
        Initialize RANSAC estimator.
        
        Args:
            max_iterations: Maximum number of iterations
            threshold: Distance threshold for inliers (in pixels)
            confidence: Desired confidence level (0-1)
        """
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.confidence = confidence
    
    def _compute_homography_dlt(self, pts1: np.ndarray, pts2: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute homography using Direct Linear Transform (DLT).
        
        Args:
            pts1: Source points (Nx2)
            pts2: Destination points (Nx2)
            
        Returns:
            3x3 homography matrix or None if computation failed
        """
        n = pts1.shape[0]
        
        if n < 4:
            return None
        
        # Build the A matrix for DLT
        A = np.zeros((2 * n, 9))
        
        for i in range(n):
            x, y = pts1[i]
            u, v = pts2[i]
            
            A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
            A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        
        return H
    
    def _compute_reprojection_error(
        self, H: np.ndarray, pts1: np.ndarray, pts2: np.ndarray
    ) -> np.ndarray:
        """
        Compute reprojection errors for all point pairs.
        
        Args:
            H: 3x3 homography matrix
            pts1: Source points (Nx2)
            pts2: Destination points (Nx2)
            
        Returns:
            Array of reprojection errors for each point
        """
        # Convert to homogeneous coordinates
        ones = np.ones((pts1.shape[0], 1))
        pts1_h = np.hstack([pts1, ones])
        
        # Project using H
        pts2_projected_h = (H @ pts1_h.T).T
        pts2_projected = pts2_projected_h[:, :2] / pts2_projected_h[:, 2:3]
        
        # Compute errors
        errors = np.linalg.norm(pts2 - pts2_projected, axis=1)
        
        return errors
    
    def estimate(
        self, pts1: np.ndarray, pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Estimate robust homography using RANSAC.
        
        Args:
            pts1: Source points (Nx2)
            pts2: Destination points (Nx2)
            
        Returns:
            (homography, inlier_mask) where:
                - homography: 3x3 matrix or None if estimation failed
                - inlier_mask: Boolean array indicating inliers
        """
        n = pts1.shape[0]
        
        if n < 4:
            return None, np.array([])
        
        best_H = None
        best_inliers = 0
        best_mask = np.zeros(n, dtype=bool)
        
        # Compute number of iterations needed
        # N = log(1-confidence) / log(1 - w^4), where w is expected inlier ratio
        # Use heuristic: estimate w from first fit
        sample_H = self._compute_homography_dlt(pts1[:4], pts2[:4])
        if sample_H is not None:
            errors = self._compute_reprojection_error(sample_H, pts1, pts2)
            inlier_ratio = np.sum(errors < self.threshold) / n
            if inlier_ratio > 0.01:  # Avoid log(0)
                iterations = int(np.log(1 - self.confidence) / 
                               np.log(1 - inlier_ratio ** 4))
                self.max_iterations = min(self.max_iterations, iterations)
        
        # RANSAC iterations
        for iteration in range(self.max_iterations):
            # Randomly select 4 points
            sample_indices = np.random.choice(n, 4, replace=False)
            sample_pts1 = pts1[sample_indices]
            sample_pts2 = pts2[sample_indices]
            
            # Compute homography
            H = self._compute_homography_dlt(sample_pts1, sample_pts2)
            
            if H is None:
                continue
            
            # Compute errors
            errors = self._compute_reprojection_error(H, pts1, pts2)
            
            # Find inliers
            inlier_mask = errors < self.threshold
            n_inliers = np.sum(inlier_mask)
            
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_H = H
                best_mask = inlier_mask
        
        # Refine with all inliers
        if best_inliers > 0:
            inlier_pts1 = pts1[best_mask]
            inlier_pts2 = pts2[best_mask]
            best_H = self._compute_homography_dlt(inlier_pts1, inlier_pts2)
        
        return best_H, best_mask


def match_features(
    kps1: List,
    des1: np.ndarray,
    kps2: List,
    des2: np.ndarray,
    ratio_threshold: float = 0.75,
) -> List[Tuple[int, int]]:
    """
    Match SIFT features using Lowe's ratio test.
    
    Args:
        kps1, des1: Keypoints and descriptors from first image
        kps2, des2: Keypoints and descriptors from second image
        ratio_threshold: Lowe's ratio threshold (0.75 is standard)
        
    Returns:
        List of (idx1, idx2) pairs for matched keypoints
    """
    if des1.size == 0 or des2.size == 0:
        return []
    
    matches = []
    
    # Convert to float32 for accurate distance computation
    des1_float = des1.astype(np.float32)
    des2_float = des2.astype(np.float32)
    
    # Brute force matching with L2 distance
    for i in range(des1.shape[0]):
        # Compute Euclidean distances to all descriptors in image 2
        distances = np.linalg.norm(des1_float[i] - des2_float, axis=1)
        
        # Get two nearest neighbors
        if len(distances) < 2:
            continue
            
        nearest_indices = np.argsort(distances)
        nearest = nearest_indices[:2]
        
        # Apply Lowe's ratio test
        if len(nearest) == 2 and distances[nearest[1]] > 0:
            if distances[nearest[0]] < ratio_threshold * distances[nearest[1]]:
                matches.append((i, nearest[0]))
    
    return matches

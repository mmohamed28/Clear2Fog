import os
import logging
import numpy as np
import multiprocessing as mp
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from .LiDAR_fog_sim import fog_simulation
from .LiDAR_fog_sim import generate_integral_lookup_table


class FogLidar:
    def __init__(self):
        """
        Initialises FogLidar and saves the path to the lookup tables for fog simulation.
        """
        self.lookup_table_dir = Path("./src/fog_pipeline/simulate_fog/lidar/LiDAR_fog_sim/integral_lookup_tables")

    def check_and_generate_lookup(self, alpha, visibility):
        """
        Checks if a lookup table for a given alpha exists. If not, it generates it.

        The provided default visibility levels: 15m, 30m, 50m, 100m and 200m.

        Args:
            alpha (float):
                The atmospheric extinction coefficient (1/m).
            visibility (float): 
                The desired meteorological optical range (MOR) in metres.
        """
        filename = f'integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_{alpha:.3f}.pickle'
        filepath = self.lookup_table_dir / filename

        if not filepath.exists():
            print("\n")
            logging.info(f"Lookup table for visibility={visibility}m (alpha={alpha:.3f}) not found. Generating look up...")
            
            # Create a simple object to pass arguments to the generation function
            class GenArgs:
                def __init__(self, save_path):
                    self.alphas = [alpha]
                    self.n_cpus = mp.cpu_count()
                    self.r_0_max = 200
                    self.n_steps = 10 * self.r_0_max 
                    self.shift = True
                    self.save_path = save_path

            args = GenArgs(self.lookup_table_dir)

            generate_integral_lookup_table.generate_integral_lookup_tables(args)
            logging.info(f"Generation complete. File saved to {filepath}")
            print("\n")

    def simulate(self, point_cloud, visibility, max_range_for_vis=None):
        """
        Applies fog simulation to a LiDAR point cloud for a given visibility.

        Args:
            point_cloud (np.ndarray): 
                Input point cloud as a NumPy array of shape (N, 4), where each row is [x, y, z, intensity]. The intensity values must be in the range [0, 255].
            visibility (float): 
                The desired meteorological optical range (MOR) in metres.
            max_range_for_vis (float, optional):
                If provided, filters the output point cloud to this maximum range (default: None).

        Returns:
            foggy_point_cloud (np.ndarray): 
                The fog-augmented point cloud as a NumPy array of shape (M, 4), where each row is [x, y, z, intensity] and M <= N. Intensities are attenuated according to fog.
        """
        alpha = np.log(20) / visibility
        self.check_and_generate_lookup(alpha, visibility)

        parameter_set = fog_simulation.ParameterSet(alpha=alpha)
        foggy_point_cloud, _, _= fog_simulation.simulate_fog(parameter_set, point_cloud, noise=10)

        # If a max range is specified, apply the filter
        if max_range_for_vis is not None:
            ranges = np.linalg.norm(foggy_point_cloud[:, :3], axis=1)
            mask = ranges <= max_range_for_vis
            return foggy_point_cloud[mask]
        
        # Otherwise, return the full, unfiltered point cloud
        return foggy_point_cloud

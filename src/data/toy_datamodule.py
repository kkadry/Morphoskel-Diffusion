import numpy as np
from scipy.interpolate import interp1d
import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


class CoronaryArteryDatasetGenerator:
    def __init__(self, voxel_grid_size, config, config_bif, num_bif_max):
        """
        Initializes the dataset generator with voxel grid size.

        :param voxel_grid_size: Tuple of (C, H, W, D) dimensions.
        """
        self.C, self.H, self.W, self.D = voxel_grid_size
        self.config = config
        self.config_bif = config_bif
        self.num_bif_max = num_bif_max

    def sample_radial_values(self, min_val, max_val, num_spline_points):
        """
        Samples radial values between min_val and max_val using the number of spline points.

        :param min_val: Minimum value.
        :param max_val: Maximum value.
        :param num_spline_points: Number of spline points for radial value sampling.
        :return: Array of sampled values.
        """
        return np.random.uniform(min_val, max_val, num_spline_points)

    def interpolate_values(self, sampled_values, voxel_depth):
        """
        Interpolates sampled values to match voxel grid depth.

        :param sampled_values: Array of sampled values.
        :param voxel_depth: Number of points along the z-axis.
        :return: Interpolated values as a 1D array.
        """
        original_positions = np.linspace(0, 1, len(sampled_values))
        target_positions = np.linspace(0, 1, voxel_depth)
        interpolator = interp1d(original_positions, sampled_values, kind="cubic")
        return interpolator(target_positions)

    def generate_labelmap(self, config):
        """
        Generates the BCHWD labelmap for the coronary artery.

        :param config: Dictionary containing configuration parameters.
        :return: 4D numpy array representing the labelmap.
        """
        labelmap = np.zeros((self.C, self.H, self.W, self.D), dtype=np.uint8)

        y, x = np.ogrid[: self.H, : self.W]
        center_y_base = self.H / 2
        center_x_base = self.W / 2

        # Sample and interpolate center_r and center_theta offsets
        center_r_samples = self.sample_radial_values(
            config["center_r_min"], config["center_r_max"], config["num_spline_points"]
        )
        center_r_offsets = self.interpolate_values(center_r_samples, self.D)

        center_theta_samples = self.sample_radial_values(
            config["center_theta_min"],
            config["center_theta_max"],
            config["num_spline_points"],
        )
        center_theta_offsets = self.interpolate_values(center_theta_samples, self.D)

        # Integrate offsets to get actual center positions and clamp to 0.5 the voxel grid dimensions
        center_r_positions = np.cumsum(center_r_offsets)
        center_r_positions = np.clip(
            center_r_positions, 0, np.sqrt((self.W / 4) ** 2 + (self.H / 4) ** 2)
        )

        center_theta_positions = np.cumsum(center_theta_offsets)
        random_shift = np.random.uniform(0, 360)
        center_theta_positions = (
            center_theta_positions + random_shift
        ) % 360  # Convert to 0-360 degrees with random shift
        center_theta_positions = np.clip(
            center_theta_positions, 0, 180
        )  # Clamp to 0-180 degrees

        z_start = int(config["z_start"] * self.D)
        z_len = int(config["z_len"] * self.D)
        z_end = z_start + z_len

        # Crop center_r_positions and center_theta_positions
        center_r_positions = center_r_positions[:z_len]
        center_theta_positions = center_theta_positions[:z_len]

        for c in range(self.C):
            # Sample and interpolate lumen radii
            lumen_radii_samples = self.sample_radial_values(
                config["r_min"], config["r_max"], config["num_spline_points"]
            )
            lumen_radii = self.interpolate_values(lumen_radii_samples, self.D)

            # Sample and interpolate vessel wall thickness
            vessel_thickness_samples = self.sample_radial_values(
                config["vessel_thickness_min"],
                config["vessel_thickness_max"],
                config["num_spline_points"],
            )
            vessel_thickness = self.interpolate_values(vessel_thickness_samples, self.D)

            r_wall = lumen_radii + vessel_thickness

            # Sample and interpolate ang_start and arc_length for calcium
            ang_start_samples = self.sample_radial_values(
                config["ang_min"], config["ang_max"], config["num_spline_points"]
            )
            ang_start_values = self.interpolate_values(ang_start_samples, self.D)

            arc_length_samples = self.sample_radial_values(
                config["arc_min"], config["arc_max"], config["num_spline_points"]
            )
            arc_length_values = self.interpolate_values(arc_length_samples, self.D)

            # Sample and interpolate calcium thickness proportions
            calcium_thickness_samples = self.sample_radial_values(
                config["calcium_thickness_min"],
                config["calcium_thickness_max"],
                config["num_spline_points"],
            )
            calcium_thickness = self.interpolate_values(
                calcium_thickness_samples, self.D
            )

            # Crop all interpolated values
            lumen_radii = lumen_radii[z_start:z_end]
            vessel_thickness = vessel_thickness[z_start:z_end]
            r_wall = r_wall[z_start:z_end]
            ang_start_values = ang_start_values[z_start:z_end]
            arc_length_values = arc_length_values[z_start:z_end]
            calcium_thickness = calcium_thickness[z_start:z_end]

            for z in range(z_len):
                center_x = center_x_base + center_r_positions[z] * np.cos(
                    np.radians(center_theta_positions[z])
                )
                center_y = center_y_base + center_r_positions[z] * np.sin(
                    np.radians(center_theta_positions[z])
                )

                distance_grid = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                angle_grid = (
                    np.degrees(np.arctan2(y - center_y, x - center_x)) + 360
                ) % 360

                labelmap[c, distance_grid <= lumen_radii[z], z + z_start] = 1  # Lumen
                mask = (distance_grid > lumen_radii[z]) & (distance_grid <= r_wall[z])
                labelmap[c, mask, z + z_start] = 3  # Vessel wall

                ang_start = ang_start_values[z]
                arc_length = arc_length_values[z]
                ang_end = (ang_start + arc_length) % 360

                if ang_start < ang_end:
                    mask_angle = (angle_grid >= ang_start) & (angle_grid < ang_end)
                else:
                    mask_angle = (angle_grid >= ang_start) | (angle_grid < ang_end)

                cal_prop = calcium_thickness[z]
                cal_thickness = vessel_thickness[z] * cal_prop
                inner_cal = lumen_radii[z] + (vessel_thickness[z] - cal_thickness) / 2
                outer_cal = lumen_radii[z] + (vessel_thickness[z] + cal_thickness) / 2

                calcium_mask = (
                    mask_angle
                    & (distance_grid >= inner_cal)
                    & (distance_grid <= outer_cal)
                )
                labelmap[c, calcium_mask, z + z_start] = 2  # Calcium

        return labelmap

    def combine_labelmaps(self, labelmaps):
        """
        Combines multiple labelmaps into a single labelmap.
        The order of resolution is: lumen > vessel wall > calcium > background.

        :param labelmaps: List of labelmaps to combine
        :return: Combined labelmap
        """
        if not labelmaps:
            return None

        combined = np.zeros_like(labelmaps[0])

        # Define priority of labels
        priority = {
            1: 3,
            3: 2,
            2: 1,
            0: 0,
        }  # 1: lumen, 3: vessel wall, 2: calcium, 0: background

        for labelmap in labelmaps:
            # For each voxel, keep the label with the highest priority
            mask = np.vectorize(priority.get)(labelmap) > np.vectorize(priority.get)(
                combined
            )
            combined[mask] = labelmap[mask]

        return combined

    def __getitem__(self, idx):
        """
        Creates a dataset of labelmaps.

        :param num_samples: Number of labelmaps to generate.
        :param config: Dictionary containing configuration parameters.
        :param config_bif: Dictionary containing bifurcation configuration parameters.
        :return: Numpy array of shape (num_samples, C, H, W, D).
        """
        num_bif = np.random.randint(0, self.num_bif_max)
        labelmap = self.generate_labelmap(self.config)
        for _ in range(num_bif):
            # Randomly sample z_start
            self.config_bif["z_start"] = np.random.uniform(
                0, 1 - self.config_bif["z_len"]
            )
            labelmap_bif = self.generate_labelmap(self.config_bif)
            labelmap = self.combine_labelmaps([labelmap, labelmap_bif])

        labelmap = torch.tensor(labelmap)
        # One hot encode the labelmap
        labelmap_oh = torch.nn.functional.one_hot(
            labelmap[0].long(), num_classes=4
        ).permute(3, 0, 1, 2)
        batch = {"label": {"data": labelmap_oh}}
        return batch

    def __len__(self):
        return 100


class CoronaryToyDataModule(nn.Module):
    def __init__(
        self, voxel_grid_size, config, config_bif_diff, num_bif_max, batch_size
    ):
        super().__init__()
        self.config = config

        self.config_bif_diff = config_bif_diff
        self.config_bif = copy.deepcopy(config)
        self.config_bif.update(config_bif_diff)

        self.num_bif_max = num_bif_max
        self.voxel_grid_size = voxel_grid_size
        self.batch_size = batch_size
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.dataset = CoronaryArteryDatasetGenerator(
            self.voxel_grid_size, self.config, self.config_bif, self.num_bif_max
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

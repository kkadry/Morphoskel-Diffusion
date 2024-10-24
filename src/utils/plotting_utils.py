# %%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import torch


class AnatomicPlotter(torch.nn.Module):
    # slice_img
    def fig2img(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image = torch.tensor(np.array(Image.open(buf).convert("RGB")))
        plt.clf()
        return image

    def slice_img(self, x):
        shape = x.shape
        # Slices into the artery with 4 cuts, assuming onehot x is in bchwd form
        x_proc = torch.argmax(x, 1).unsqueeze(0)
        slice_long_x = x_proc[0, 0, ..., int(shape[2] / 2), :, :].detach().cpu()
        slice_long_y = x_proc[0, 0, ..., int(shape[3] / 2), :].detach().cpu()
        slice_cross_z = x_proc[0, 0, ..., int(shape[4] / 2)].detach().cpu()
        return (slice_long_x, slice_long_y, slice_cross_z)

    def depthmap(self, x):
        # Plot depth maps of binary image, assuming bchwd
        x_flat_x = torch.argmax(x, 2)[0, 0].detach().cpu()
        x_flat_y = torch.argmax(x, 3)[0, 0].detach().cpu()
        x_flat_z = torch.argmax(x, 4)[0, 0].detach().cpu()
        return (x_flat_x, x_flat_y, x_flat_z)

    # Plots 4 subplots of the generated geometry
    def plot_4views(self, img, plotcheck=True):
        slice_list = self.slice_img(img)
        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(slice_list[0])
        axs[1].imshow(slice_list[1])
        axs[2].imshow(slice_list[2])
        # removing axis ticks
        for ax in axs.flat:
            ax.axis("off")
        plt.show()
        plt.tight_layout()
        img_fig = self.fig2img(fig)
        return img_fig, axs, slice_list

    def plot_morph_comparison(self, metrics, sample_index=0):
        batch_metrics = metrics["anatomic_batch_metrics"]
        # Extracting out conditional metrics
        cond_metrics = batch_metrics.filter_by_attrs(cond=True)
        cond_metrics_morph = cond_metrics.filter_by_attrs(
            metric_type="morph", metric_dim=2
        )

        morph_sample = cond_metrics_morph.filter_by_attrs(source="sample")
        morph_seed = cond_metrics_morph.filter_by_attrs(source="seed")

        # Get the number of data variables in cond_metrics_morph_sample
        n = len(morph_sample.data_vars)  # Subplots
        fig, axs = plt.subplots(n)
        ax_list = axs.flatten()

        for v, (sample_var, seed_var) in enumerate(
            zip(morph_sample.data_vars, morph_seed.data_vars)
        ):
            sample_values = morph_sample[sample_var].values.flatten()[
                128 * sample_index : 128 * (sample_index + 1)
            ]
            seed_values = morph_seed[seed_var].values.flatten()[
                128 * sample_index : 128 * (sample_index + 1)
            ]
            ax_list[v].plot(seed_values, label="seed")
            ax_list[v].plot(sample_values, label="sample")

            # Add title
            ax_list[v].set_xlabel(sample_var)
            # Remove xticks
            ax_list[v].set_xticks([])
        # Put legend outside figure, at middle, make it based on labels
        handles, labels = ax_list[0].get_legend_handles_labels()
        fig.legend(
            handles,
            ["Seed", "Sample"],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
        )

        return fig

    def plot_3d_skeleton(self, pts_sample, pts_seed, subsamp=1, title=None):
        # Create a 3D scatter plot for pts_sample and pts_seed
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # permute pts_sample and pts_seed to be z,y,x
        pts_sample = pts_sample[::subsamp, [2, 1, 0]]
        pts_seed = pts_seed[::subsamp, [2, 1, 0]]
        # flip all axes
        pts_sample = 128 - pts_sample
        pts_seed = 128 - pts_seed
        # Extract x, y, and z coordinates for sample points
        x_sample = pts_sample[:, 0]
        y_sample = pts_sample[:, 1]
        z_sample = pts_sample[:, 2]

        # Extract x, y, and z coordinates for seed points
        x_seed = pts_seed[:, 0]
        y_seed = pts_seed[:, 1]
        z_seed = pts_seed[:, 2]

        # Create the scatter plot for sample points (red)
        ax.scatter(x_sample, y_sample, z_sample, c="red", label="Sample Skel.")

        # Create the scatter plot for seed points (blue)
        ax.scatter(x_seed, y_seed, z_seed, c="blue", label="Seed Skel.")

        # Set labels and title
        ax.set_xlabel("Longitudial Direction")

        # Add a legend
        ax.legend()

        # Set xyz lim to be 10-118
        ax.set_xlim(10, 118)
        ax.set_ylim(10, 118)
        ax.set_zlim(10, 118)

        # Show the plot
        plt.show()
        return fig

    def plot_skeleton_comparison(self, metrics, sample_index=0, subsample=5):
        batch_metrics = metrics["anatomic_batch_metrics"]
        cond_metrics_skel = batch_metrics.filter_by_attrs(metric_type="skel")
        skel_sample = cond_metrics_skel.filter_by_attrs(source="sample")
        skel_seed = cond_metrics_skel.filter_by_attrs(source="seed")

        # Plotting skeleton depth map
        pts_sample = skel_sample["cond_skel_sample_lumen_hard_skel"].values[
            sample_index
        ]
        pts_seed = skel_seed["cond_skel_seed_lumen_hard_skel"].values[sample_index]

        # Example usage:
        fig = self.plot_3d_skeleton(
            pts_seed, pts_sample, subsamp=subsample, title="Seed and Sample Points"
        )
        return fig

    def plot_seed_sample_comparison(self, x_hat, seed):
        slices_seed, _, _ = self.plot_4views(seed)
        slices_sample, _, _ = self.plot_4views(x_hat)

        fig, axs = plt.subplots(2, 1, facecolor="white")
        fig.patch.set_facecolor("white")

        # Seed
        axs[0].imshow(slices_seed)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title("Seed Label Map")

        # Sample
        axs[1].imshow(slices_sample)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_title("Sample Label Map")
        for ax in axs:
            ax.set_facecolor("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

        plt.tight_layout()
        return fig

    def log_anatomic_plots(self, metrics, x_hat, seed):
        sample_index = 0
        fig_morph = self.plot_morph_comparison(
            metrics=metrics, sample_index=sample_index
        )
        img_morph = self.fig2img(fig_morph)

        fig_skel = self.plot_skeleton_comparison(
            metrics=metrics, sample_index=sample_index, subsample=5
        )
        img_skel = self.fig2img(fig_skel)

        fig_seed_sample = self.plot_seed_sample_comparison(x_hat, seed)
        img_seed_sample = self.fig2img(fig_seed_sample)

        figs = {
            "fig_morph": fig_morph,
            "fig_skel": fig_skel,
            "fig_seed_sample": fig_seed_sample,
        }
        imgs = {
            "img_morph": img_morph,
            "img_skel": img_skel,
            "img_seed_sample": img_seed_sample,
        }
        anatomic_plots = {"figs": figs, "imgs": imgs}
        return anatomic_plots

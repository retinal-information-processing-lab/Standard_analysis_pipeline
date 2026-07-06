import numpy as np
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from math import *

# Above is just everything from original utils.  Likely overkill but prevents annoying errors from not having a needed module.


import utils
import params


def import_data_to_plot(params: dict):
    """Load all data required for generating ID card plots.

    Args:
        params (dict): Experiment parameters from params.py containing:
            - exp (str): Experiment name
            - output_directory (str): Path to analysis output directory
            - phy_directory (str): Path to Phy output directory

    Returns:
        tuple: Contains:
            - exp (str): Experiment name
            - output_directory (str): Output directory path
            - phy_directory (str): Phy directory path
            - check_directory (str): Checkerboard analysis directory
            - DG_directory (str): Direction selectivity directory
            - CT_directory (str): Cell typing directory
            - euler_vec (np.ndarray): Chirp stimulus vector
            - check_rast (dict): Checkerboard raster data
            - DG_data (dict): Direction selectivity data
            - sta_results (dict): STA analysis results
            - cells (list): List of cell IDs
            - Chirp_data (dict): Chirp response data

    Note:
        This function loads all necessary data for creating comprehensive
        cell ID cards. Consider refactoring to return a single dict/object
        for cleaner code organization.
    """
    # Extract parameters
    exp = params.exp
    output_directory = params.output_directory
    phy_directory = params.phy_directory

    # Find analysis directories
    check_directory = utils.find_analysis_directory(
        output_directory, dir_type="Checkerboard"
    )
    DG_directory = utils.find_analysis_directory(output_directory, dir_type="DG")
    CT_directory = utils.find_analysis_directory(
        output_directory, dir_type="CellTyping"
    )

    # Load data
    # load chirp stimulus for plotting the profile
    vec_path = os.path.join(
        params.stim_directory, r"Euler_50Hz_20reps_1024x768pix.vec"
    )
    euler_vec = np.genfromtxt(vec_path)

    # load the checkerboard repeated-sequence rasters (notebook 2 saves them via
    # save_obj, i.e. as "Check_rasters_data.pkl", not a standalone .npy anymore)
    if not os.path.isfile(os.path.join(check_directory, "Check_rasters_data.pkl")):
        print(
            "Checkerboard rasters not found. If you did not use SWAN, please run the checkerboard analysis (notebook 2) first."
        )
        check_rast = None
    else:
        check_rast = utils.load_obj(os.path.join(check_directory, "Check_rasters_data.pkl"))

    # load the DG data
    DG_data = np.load(
        os.path.join(DG_directory, "DG_data_exp{}.pkl".format(exp)), allow_pickle=True
    )

    # load the STA analysis results from the standard checkerboard analysis (notebook 2):
    # prefer the extended version (physical units), fall back to the plain analysed one.
    sta_file = os.path.join(check_directory, "sta_data_analysed_extended.pkl")
    if not os.path.isfile(sta_file):
        sta_file = os.path.join(check_directory, "sta_data_analysed.pkl")
    sta_results = utils.load_obj(sta_file)
    cells = list(sta_results.keys())

    # load the chirp
    Chirp_data = np.load(
        os.path.join(CT_directory, "{}_cell_typing_data.pkl".format(exp)),
        allow_pickle=True,
    )

    return (
        exp,
        output_directory,
        phy_directory,
        check_directory,
        DG_directory,
        CT_directory,
        euler_vec,
        check_rast,
        DG_data,
        sta_results,
        cells,
        Chirp_data,
    )


def select_and_save_good_cells(
    cells: list, cell_rpvs: dict, output_directory: str, rpv_threshold: float = 0.5
):
    """Select and save cells that meet quality criteria based on RPV.

    Args:
        cells (list): List of all cell IDs
        cell_rpvs (dict): Dictionary containing RPV data for each cell
        output_directory (str): Path to save good cells array
        rpv_threshold (float): Maximum acceptable RPV percentage. Default 0.5%.

    Returns:
        np.ndarray: Array of good cell IDs

    Note:
        Saves the good cells array to 'Good_cells.npy' in output_directory.
        Also prints the list of good cells to console.
    """
    good_cells = []
    for cell_nb in cells:
        if cell_rpvs[cell_nb]["rpv"] < rpv_threshold:
            good_cells.append(cell_nb)

    good_cells = np.array(good_cells)
    np.save(os.path.join(output_directory, r"Good_cells"), good_cells)
    print(good_cells)

    return good_cells


def create_id_cards_and_plots(
    cells,
    cell_rpvs,
    output_directory,
    sta_results,
    check_rast,
    Chirp_data,
    DG_data,
    euler_vec,
    exp,
    rpv_len=0.002,
    fontsize: int = 16,
    n_sigma: float = 2.0,
):
    """Generate and save ID cards for all cells.

    Args:
        cells (list): List of cell IDs to generate ID cards for
        cell_rpvs (dict): Refractory period violation data for all cells
        output_directory (str): Path to save ID cards

    Returns:
        None. Saves ID card figures to output_directory/ID_cards/

    Note:
        Creates comprehensive ID cards showing multiple stimulus responses
        and quality metrics for each cell. Handles missing data gracefully.
        RWD self-note: not clear if figures are the ID cards or ID cards and figures in this function are separate thing

    """
    # Input----------------------------------------------------------

    fig_directory = os.path.normpath(os.path.join(output_directory, r"ID_cards"))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)

    # Plotting--------------------------------------------------------

    for cell_nb in tqdm(cells[:]):
        # Create the figure
        fig = plt.figure(figsize=(10, 12))

        # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(
            7, 5, left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.7
        )

        if Chirp_data[cell_nb]["type"] == "Not assigned":
            cluster = ""
        else:
            cluster = Chirp_data[cell_nb]["type"]

        plt.suptitle(
            "exp{} _c{}  - Cluster_group_{} ".format(exp, cell_nb, cluster),
            fontsize=fontsize + 6,
        )
        # --------------------------------------------------
        # Plot the ISI
        ax = fig.add_subplot(gs[0:2, 0:1])
        ax.hist(cell_rpvs[cell_nb]["isi"] * 1000, bins=100, range=(0, 50))
        rpv = cell_rpvs[cell_nb]["rpv"]
        nb_spikes = cell_rpvs[cell_nb]["nb_spikes"]
        nb_rpv = cell_rpvs[cell_nb]["nb_rpv_spikes"]
        ax.axvline(rpv_len, lw=0.5, color="k")
        ax.set_title(
            "Interspike Interval histogram\n RPV = {}%. {}/{} spikes".format(
                round(rpv, 4), int(nb_rpv), nb_spikes
            ),
            fontsize=fontsize,
        )
        ax.set_xlabel("Interspike time (ms)", fontsize=fontsize)
        ax.set_ylabel("Number of spikes", fontsize=fontsize)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(0, color="k", lw=0.5)

        # --------------------------------------------------
        # Plot spatial STA (from the standard analysis) with the fitted ellipse, and
        # report the RF quality metrics (SNR, size, fitted) shown on the new RF plot.
        ax = fig.add_subplot(gs[0:2, 3:5])
        sta_analysis = sta_results[cell_nb]["sta_analysis"]
        ellipse = sta_analysis["EllipseCoor"]
        spatial = sta_analysis["Spatial"]
        level_factor = np.exp(
            -(n_sigma**2) / 2
        )  # peak fraction of a Gaussian at n_sigma
        utils.plot_sta(ax, spatial, ellipse, level_factor=level_factor, color="yellow")
        ax.set_xticks([])
        ax.set_yticks([])

        title = "Spatial receptive field"
        try:
            if ellipse[0] != 0 and sta_analysis.get("FittedEllipse", True):
                snr = utils.rf_snr(
                    spatial, ellipse, method="peak_std", level_factor=level_factor
                )
                if "EllipseCoor_um" in sta_analysis:
                    diameter = utils.ellipse_diameter(
                        sta_analysis["EllipseCoor_um"], method="circle_approx"
                    )
                    area = utils.ellipse_area(
                        sta_analysis["EllipseCoor_um"], method="formula"
                    )
                    size_txt = f"Ø {diameter:.0f} µm · {area:.0f} µm²"
                else:
                    diameter = utils.ellipse_diameter(ellipse, method="circle_approx")
                    area = utils.ellipse_area(ellipse, method="formula")
                    size_txt = f"Ø {diameter:.1f} · {area:.1f} unit²"
                title += f"\nSNR {snr:.1f} · {size_txt}"
            else:
                title += "\n(ellipse not fitted)"
        except Exception as err:
            print(f"Warning: RF metrics failed for cell {cell_nb} ({err}).")
        ax.set_title(title, fontsize=fontsize)

        # --------------------------------------------------
        # Plot checkerboard repeated sequence raster
        if check_rast is not None:
            ax = fig.add_subplot(gs[4:6, 3:5])
            ax.eventplot(
                check_rast[cell_nb]["spike_trains"], color="k", alpha=1, linelengths=1
            )
            ax.set_title("Repeated white noise sequences", fontsize=fontsize)
            ax.set_xlabel("Time (s)", fontsize=fontsize)
            seq_lenght = (
                check_rast[cell_nb]["repeated_sequences_times"][0][1]
                - check_rast[cell_nb]["repeated_sequences_times"][0][0]
            )
            ax.set_xlim([0, seq_lenght])
            ax.set_ylim([0, None])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # --------------------------------------------------
            # Plot checkerboard repeated sequence psth (superimposed)
            ax = fig.add_subplot(gs[6:7, 3:5])
            width = check_rast[cell_nb]["repeated_sequences_times"][0][0] / int(1200 / 2)
            seq_lenght = (
                check_rast[cell_nb]["repeated_sequences_times"][0][1]
                - check_rast[cell_nb]["repeated_sequences_times"][0][0]
            )
            ax.bar(
                np.linspace(0, seq_lenght, int(1200 / 2)) + width / 2,
                check_rast[cell_nb]["psth"],
                width=1.3 * width,
            )
            ax.set_xlabel("Time (s)", fontsize=fontsize)
            ax.set_xlim([0, seq_lenght])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # --------------------------------------------------
        # Plot temporal STA
        ax = fig.add_subplot(gs[0:2, 1:3])
        ax.set_title("Temporal receptive field", fontsize=fontsize)
        ax.step(
            np.linspace(-1, 0, 21),
            sta_results[cell_nb]["sta_analysis"]["Temporal"][-21:],
            color="k",
            lw=3,
        )
        ax.set_xlabel("Time (s)", fontsize=fontsize)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_yticks([])
        ax.axis("off")

        # --------------------------------------------------
        # Plot chirp psth
        ax = fig.add_subplot(gs[6:7, 0:3])
        ax.plot(np.linspace(0, 32, 800), Chirp_data[cell_nb]["psth"])
        ax.set_xlabel("Time (s)", fontsize=fontsize)
        ax.set_ylabel("Firing rate (spikes/s)", fontsize=fontsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 32)

        # --------------------------------------------------
        # Plot chirp raster
        ax = fig.add_subplot(gs[4:6, 0:3])
        ax.eventplot(Chirp_data[cell_nb]["spike_trains"], color="k", alpha=1)
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 20)
        ax.set_ylabel("#Trial", fontsize=fontsize)
        ax.set_title("Response to the chirp stimulus", fontsize=fontsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # --------------------------------------------------
        # Plot chirp stimulus profile (superimposed)
        ax = fig.add_subplot(gs[3:4, 0:3])
        ax.plot(
            np.linspace(0, 32, 1600),
            euler_vec[0 + 151 : 151 + 1600, 1],
            color="k",
            lw=0.75,
        )
        # ax.set_ylim(-800,300)
        ax.set_yticks([])
        #     ax.set_title("Chirp stimulus profile")
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 32)

        # --------------------------------------------------
        # Plot Polar plot
        ax = fig.add_subplot(gs[2:4, 3:5], polar=True)
        try:
            atune = DG_data[cell_nb]["atune"]
            R = DG_data[cell_nb]["Rtune"]
            TuneSum = DG_data[cell_nb]["Tuning"]
            IDX = DG_data[cell_nb]["IDX"]

            theta = np.linspace(0, 2 * np.pi, 9)
            ax.plot([atune, atune], [0, R], "b-")
            ax.plot([atune], [R], "bo")
            ax.plot(theta, TuneSum)
            ax.fill(theta, TuneSum, "b", alpha=0.1)

            ax.text(
                np.pi / 2 * 6 / 8, 2.6, "IDX = " + str(np.round(IDX, 1)), size=fontsize
            )
            ax.text(np.pi / 2 * 6 / 9, 2.2, "R = " + str(np.round(R, 1)), size=fontsize)

            ax.set_yticks([0, 0.5, 1, 1.5, 2])
            ax.set_yticklabels([0, "", 1, "", 2])

            # plot rasters of DG
            ax = fig.add_subplot(gs[2:3, 0:3])
            seq_sep = 20
            seq_len = 12
            ch_raster = DG_data[cell_nb]["rasters"]
            ax.eventplot(ch_raster[:], color="k", lw=1, linelengths=0.95)
            for a in np.arange(8):
                ax.axvline(a * seq_sep, color="gray", lw=2)
                ax.axvline(a * seq_sep + seq_len, color="gray", lw=2)
                ax.axvline(a * seq_sep + seq_len / 6, color="gray", ls="--", lw=1.5)

            ax.set_xlim([-seq_sep / 2, seq_sep * 8])
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks(
                [6, 26, 46, 66, 86, 106, 126, 146], [0, 45, 90, 135, 180, 225, 270, 315]
            )
            ax.set_title("DG rasters", fontsize=fontsize)

            fsave = os.path.join(
                fig_directory, "Group{}_cell{}".format(cluster, cell_nb)
            )

            fig.savefig(fsave + ".png", format="png", dpi=110)
            plt.close(fig)

        except:
            fsave = os.path.join(
                fig_directory, "Group{}_cell{}".format(cluster, cell_nb)
            )

            fig.savefig(fsave + ".png", format="png", dpi=110)
            plt.close(fig)

    return

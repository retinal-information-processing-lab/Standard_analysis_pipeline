import utils
import os
from tqdm import tqdm
import analyse_checkerboard as analysis

root = r'C:\Users\cboscarino\Documents\GitHub\Standard_analysis_pipeline\data\20251219_PulsingGratings_PupilSize\Analysis\Checkerboard_Analysis_rec_0'
data_filename = "sta_data_analysed.pkl"
data_fp = os.path.join(root, data_filename)
sta_data = utils.load_obj(data_fp)

cell_ids = list(sta_data.keys())  #[1, 2, 9, 10]

# Compute STA analysis
folder = os.path.join(root, 'STA_analysis_comparison')
os.makedirs(folder, exist_ok=True)
standard_sta = {}
matias_sta = {}
tom_sta = {}
for cell_id in tqdm(cell_ids, desc="Computing STAs"):
    sta = sta_data[cell_id]["sta_3D"]

    matias_sta[cell_id]= {'sta_analysis': utils.rf_analysis(sta, cell_id, method='matias')}
    tom_sta[cell_id]= {'sta_analysis': utils.rf_analysis(sta, cell_id, method='tom')}
    standard_sta[cell_id]= {'sta_analysis': utils.rf_analysis(sta, cell_id, method='standard')}

xdim, ydim = 8, 5
analysis.plot_sta_fitted_with_ellipse(
    standard_sta,
    folder,
    folder_name="Standard_STA",
    add_raster_plot=False,
    add_spatial_mask=True,
    cell_ids=cell_ids,  # list of cell ids to plot, e.g., [208, 209, 210], if None it will plot all cells
    show_figures=False,
    xdim=xdim,
    ydim=ydim,
    save_format="png"
)

analysis.plot_sta_fitted_with_ellipse(
    matias_sta,
    folder,
    folder_name="Matias_STA",
    add_raster_plot=False,
    cell_ids=cell_ids,  # list of cell ids to plot, e.g., [208, 209, 210], if None it will plot all cells
    show_figures=False,
    xdim=xdim,
    ydim=ydim,
    save_format="png"
)

analysis.plot_sta_fitted_with_ellipse(
    tom_sta,
    folder,
    folder_name="Tom_STA",
    add_raster_plot=False,
    cell_ids=cell_ids,  # list of cell ids to plot, e.g., [208, 209, 210], if None it will plot all cells
    show_figures=False,
    xdim=xdim,
    ydim=ydim,
    save_format="png"
)

# analysis.plot_one_cell_3D_spike_triggered_average(
#     sta_data,
#     max_frames_to_show=40,
#     n_frames_per_line=10,
#     fontsize=14)
#
# input("Press Enter to continue...")
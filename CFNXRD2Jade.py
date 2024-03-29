"""
File: CFNXRD2Jade
Name: Cheng-Chu Chung
----------------------------------------
TODO: Convert CFN XRD data to the data readable in Jade (10 --> 10.00 for x_axis value)
"""
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import palettable.colorbrewer.diverging as pld

# Step 1: Paste your directory
INPUT_PATH = r"D:\Research data\SSID\202311\20231114 b4567 comparison"   # <--- Give the folder directory you want to explore
OUTPUT_PATH = Path(INPUT_PATH) / 'Converted'  # <--- Give the folder directory you want to save the converted data

# Step 2: Set up your plotting parameters
# CONSTANT
FILE_TYPE = '.xy'
PLOT_LIST = [3, 0, 4, 1, 5, 2, 6, 7]                      # [] for default or [1, 7, 5, 3] index list for the index sequence you desire
SAMPLE_LABEL = ['b44-01_VTiCu_SP_Pristine',
                'b44-02_VTiCu_SP_750C30M',
                'b45-01_NbAlCu_SP_Pristine',
                'b45-02_NbAlCu_SP_500C30M',
                'b46-01_MoTiCu_SP_Pristine',
                'b46-02_MoTiCu_SP_750C30M',
                'b47-01_NbAlSc_SP_Pristine',
                'b47-02_NbAlSc_SP_900C30M']                   # [] for default or add a specific name list
OUTPUT = False                      # "True" if you want to save the converted file  # <-------------------------- Check
Y_RANGE = (-100, 500)               # Increment of ylim
PLOT_OFFSET = 800                   # Value you want to add to an offset for each curve.
FRAMELINEWIDTH = 1.5
LINEWIDTH = 2
IF_SAVE = True                    # "True" if you want to save the plots  # <----------------------------------- Check
IF_LEGEND = True                    # "True" if you want to show the legend
LEGEND_LOCATION = 'upper left'
PALETTE = pld.Spectral_4_r          # _r if you want to reverse the color sequence
CMAP = PALETTE.mpl_colormap         # .mpl_colormap attribute is a continuous, interpolated map
FILENAME_LENGTH = 20                # The length of the filename you want to show in the legend
OUTPUT_FILENAME = 'b4456'  # <----------------------------- Enter your figure title and file name

# --------------------Good luck for your data conversion!--------------------


def main():
    # Path function converts \ to / and glob method returns .xy files in a generator ---> Very powerful!
    files = Path(INPUT_PATH).glob(f'*{FILE_TYPE}')
    dictionary_of_I_and_q = convert_format(files)
    intensity_plot(dictionary_of_I_and_q)


# @pysnooper.snoop()
def convert_format(files):
    """
    :param files: generator, list of files in the directory
    :return: dict, a dictionary stores 2theta, intensity, and filename
    """
    list_dict = {'q_list': {}, 'I_list': {}, 'filename_list': {}}   # Create a dictionary to store all information
    # Import all the data
    print('Index Filename')
    for index, file_directory in enumerate(files):
        file = file_directory.resolve()  # Make the path absolute, resolving any symlinks
        filename = file.name
        if '.xy' in filename and 'Converted' not in filename:
            list_dict['filename_list'][index] = filename    # Append index and filename for the outline
            print(index, filename)  # Print all the files
            df = pd.read_table(
                INPUT_PATH + '/' + filename, header=None
                )
            x = np.array(df[0].tolist())   # q
            y = np.array(df[1].tolist())   # I(q)
            list_dict['q_list'][index] = x
            list_dict['I_list'][index] = y
            # print(df)
            if OUTPUT:
                out_file(x, y, f'Converted_{filename}')
    return list_dict


def out_file(tth, intensity, filename):
    """
    :param tth: Array, an array stores 2theta
    :param intensity: Array, an array stores intensity
    :param filename: List, a list stores filenames
    :return: None
    """
    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir()    # Create an output folder to save all generated data/files

    print('=================================================================================')
    print(f'Converting CFN XRD data to --> {filename}')
    # input_path = Path(INPUT_PATH)
    filename = OUTPUT_PATH / filename
    with filename.open(mode='w') as out:
        out.write('tth intensity\n')
        for angle, signal in zip(tth, intensity):
            out.write(f"{angle:.2f} {signal:.5f}\n")    # ChatGPT
    print('=================================================================================')
    print(' ')


def intensity_plot(dictionary_of_I_and_q):
    """
    :param dictionary_of_I_and_q: dict, a dictionary contains an intensity, q, and filename list
    :return: None
    """
    # Import the data
    if len(PLOT_LIST) == 0:
        index = list(np.arange(len(dictionary_of_I_and_q['filename_list'])))    # Select the index from the list_dict['filename_list']
    else:
        index = PLOT_LIST
    plot_sequence = 0
    print('Plot:')
    fig, ax = plt.subplots()
    for i in index:
        color_idx = np.linspace(0, 1, len(index))

        x = dictionary_of_I_and_q['q_list'][i]
        y = dictionary_of_I_and_q['I_list'][i] + plot_sequence*PLOT_OFFSET

        # Give specific labels
        if len(SAMPLE_LABEL) == 0:
            sample_name = dictionary_of_I_and_q['filename_list'][i][:FILENAME_LENGTH]
        else:
            sample_name = SAMPLE_LABEL[PLOT_LIST.index(i)]

        print(i, dictionary_of_I_and_q['filename_list'][i])
        plt.plot(x, y, linewidth=LINEWIDTH, color=CMAP(color_idx[plot_sequence]), label=f'{sample_name}')
        plot_sequence += 1

    # Plotting format
    # Outer frame edge width
    spineline = ['left', 'right', 'top', 'bottom']
    for direction in spineline:
        ax.spines[direction].set_linewidth(FRAMELINEWIDTH)

    x_label = r'$\mathregular{2\theta \ (degree)}$'
    y_label = r'Intensity (arb. units)'
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    plt.yticks([])  # Disable ticks
    plt.xticks(fontsize=14)
    ax.tick_params(width=FRAMELINEWIDTH)

    plt.xlim(10, 80)
    y_limit_max = dictionary_of_I_and_q['I_list'][index[-1]].max() + plot_sequence * PLOT_OFFSET + Y_RANGE[1]
    y_limit_min = dictionary_of_I_and_q['I_list'][index[0]].min() + Y_RANGE[0]
    plt.ylim(y_limit_min, y_limit_max)

    if IF_LEGEND:
        plt.legend(loc=LEGEND_LOCATION, framealpha=1, frameon=False, fontsize=14)
    plt.title(OUTPUT_FILENAME, fontsize=20, pad=10)
    plt.tight_layout()
    if IF_SAVE:
        if not OUTPUT_PATH.exists():
            OUTPUT_PATH.mkdir()    # Create an output folder to save all generated data/files
        output_filename = check_filename_repetition(OUTPUT_FILENAME)
        plt.savefig("{}/{}.png".format(Path(OUTPUT_PATH), output_filename), dpi=300, transparent=False)
    plt.show()


def check_filename_repetition(output_filename):
    """
    :param output_filename: string, output filename
    :return: string, new output filename
    """
    print("\n==============================")
    print('Check filename repetition')
    print("------------------------------")
    files = Path(INPUT_PATH).glob(f'*.png')
    png_list = []
    for index, file in enumerate(files):
        png_list.append(file.name[:-4])

    print(output_filename)
    while output_filename in png_list:
        output_filename = output_filename + '_1'
        print(output_filename)
    return output_filename


if __name__ == '__main__':
    main()

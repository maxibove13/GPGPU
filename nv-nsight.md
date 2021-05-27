# NVIDIA Nsight Compute

NVIDIA Nsight Compute is an interactive kernel profiler for CUDA applications. It provides detailed performance metrics and API debugging via a user interface and command line tool. In addition, its baseline feature allows users to compare results within the tool. NVIDIA Nsight Compute provides a customizable and data-driven user interface and metric collection and can be extended with analysis scripts for post-processing results.

## Installation guide (Linux)

1. Download the package from [NVIDIA](https://developer.nvidia.com/gameworksdownload#?dn=nsight-compute-2021-1-1) site.
2. Give permissions to the downloaded installation file:

   `chmod +x nsight*.run`

3. Run it as root or in sudo mode and follow the prompt messages:

   `sudo ./nsigh*.run`

4. In order to run it from any dir, add the root NVIDIA Nsigh Compute path to your PATH environmental variable:

   1. Open your bashrc file (just a set of commands that the terminal executes at start up):

      `nano ~/.bashrc`

   2. At the end of your bashrc add the following lines to add nv-sight path to PATH:

      `export PATH=$PATH:/usr/local/NVIDIA-Nsight-Compute`

5. Now you can just open the program by typing in the terminal:

   `ncu-ui`

## Usage

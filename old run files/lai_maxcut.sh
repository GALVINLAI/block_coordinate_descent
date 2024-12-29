#!/bin/bash

# My computer is too slow, so I can only reduce the number of repeats; otherwise, it takes too long. Default repeat=10
repeat=5
num_iter=200

# Use the seq command to generate a sequence from 0.01 to 0.2 with a step size of 0.01, and store the result in an array
# sigma_values=($(seq 0.00 0.02 0.2))
 
# sigma_values=(0.0 0.01)
# sigma_values=(0.0 0.01 0.05 0.1 0.2 0.3)

python run_maxcut.py --repeat ${repeat} --lr_gd 0.05 --lr_rcd 0.1 --num_iter ${num_iter}
python lai_plot_script.py --phys maxcut --x_lim_coff 1.1

# python big_image.py --root_dir "plots/maxcut/lr_0.1/dim_20" --output_dir "plots/maxcut/lr_0.1/dim_20/combined_images"

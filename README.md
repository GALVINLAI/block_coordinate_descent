# block_coordinate_descent

block_coordinate_descent code is `algo.bcd`

Run `lai_maxcut.sh` `lai_tsp.sh` (try to adjust the parameters `num_iter` and `sigma_values`) to show the performance of `algo.bcd` and `algo.gd` and `algo.rcd`. Then, we finally get the `exp`, `plots`. 

We have added two prerun results in this repo, which are:
 - `plots\maxcut\lr_0.1\dim_20\combined_images\combined_energy_HM_fun_evals.png`
 - `plots\tsp\lr_0.0001\dim_90\combined_images\combined_energy_HM_fun_evals.png`

Be carefull! Every time you run `lai_maxcut.sh` or `lai_tsp.sh`, the old `exp`, `plots` will be deleted.

**Notice that, for now, bcd cannot solve `tfim`, `factor`, `heisenberge`.**

Many try and tests are in `lai_playground` folder.



<p align="center">
  <a href="https://ai4life.eurobioimaging.eu/open-calls/">
    <img src="https://github.com/ai4life-opencalls/.github/blob/main/AI4Life_banner_giraffe_nodes_OC.png?raw=true" width="70%">
  </a>
</p>


# Project #11: Automated species annotation in turbid waters
---

[![DOI](https://zenodo.org/badge/823104605.svg)](https://doi.org/10.5281/zenodo.15913054)
![Static Badge](https://img.shields.io/badge/Data_on_BIA-S--BIAD2005-blue?style=flat&link=https%3A%2F%2Fdoi.org%2F10.6019%2FS-BIAD2005)

This repo was created by the [AI4Life project](https://ai4life.eurobioimaging.eu) using data provided by Nils Jacobsen at [Royal Belgian Institute of Natural Sciences](https://www.naturalsciences.be/en).
All the images demonstrated in this tutorial are provided under **CC-BY** licence.

If any of the instructions are not working, please [open an issue](https://github.com/ai4life-opencalls/oc_2_project_11/issues) or contact us at [ai4life@fht.org](ai4life@fht.org)!


## Installation
You can clone the repository and create an environment from the environment file included as follows:

```bash
git clone git@github.com:ai4life-opencalls/oc_2_project_11.git
cd oc_2_project_11
```
If you have a **GPU**:
```bash
conda env create -f env_gpu.yml
```
Otherwise:
```bash
conda env create -f env_cpu.yml
```

#### Note:
GPU installation might get a bit tricky depend on your nvidia driver and cuda version. `sam-2` on GPU needs `cuda >= 12`, otherwise it might replace the `pytorch` with the CPU version.  
You can check the `pytorch` instructions for the manual installation [here](https://pytorch.org/get-started/previous-versions/#v231).


## Usage
Once the environment is created, you can open the prompt, activate the environment and start jupyter lab.
```bash
conda activate ai4life
jupyter lab
```

Inside the `notebooks` folder you will find notebooks for:

- [BIIGLE to COCO format mask export](notebooks/biigle_sam2_coco.ipynb)
- [Paparazzi to COCO format mask export](notebooks/paparazzi_sam2_coco.ipynb)
- [Automatic mask generation](notebooks/automatic_mask_generator.ipynb)

- [BIIGLE point to polygon generation](notebooks/BIIGLE_point_to_polygon_with_sam.ipynb)

- [Paparazzi point to polygon generation](notebooks/paparazzi_point_to_polygon_with_sam.ipynb)

- [Plotting and saving masks for many images](notebooks/Plot_masks_with_sam-batch.ipynb)


<br><br>

# Acknowledgements
AI4Life has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement number 101057970. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## Contact

[SciLifeLab BioImage Informatics Facility (BIIF)](https://www.scilifelab.se/units/bioimage-informatics/)

Developed by [Mehdi Seifi](mailto:mehdi.seifi@fht.org), [Agustin Corbat and Kristina Lidayova](mailto:biif@scilifelab.se)

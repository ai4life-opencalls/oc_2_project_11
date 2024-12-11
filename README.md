<p align="center">
  <a href="https://ai4life.eurobioimaging.eu/open-calls/">
    <img src="https://github.com/ai4life-opencalls/.github/blob/main/AI4Life_banner_giraffe_nodes_OC.png?raw=true" width="70%">
  </a>
</p>


# Project #11: Automated species annotation in turbid waters


[![DOI](https://zenodo.org/badge/DOI/)](https://doi.org/)


---
This page was created by the [AI4Life project](https://ai4life.eurobioimaging.eu) using data provided by Nils Jacobsen at [Royal Belgian Institute of Natural Sciences](https://www.naturalsciences.be/en).
All the images demonstrated in this tutorial are provided under **CC-BY** licence.

If any of the instructions are not working, please [open an issue](https://github.com/ai4life-opencalls/oc_2_project_11/issues) or contact us at [ai4life@fht.org](ai4life@fht.org)!

**Project challenges**: instance segmentation

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Conclusion](#conclusion)


# Introduction

Monitoring of ecologically important marine habitats (gravel beds) and benthic communities by analysis of videos recorded by a towed underwater video system.

[Project description](https://biifsweden.github.io/projects/2024/07/02/AI4Life_OC2_2024_11/)

[Download code](archive/refs/heads/main.zip)

## Installation

You can clone the repository and create an environment from the environment file included as follows.

```
git clone git@github.com:BIIFSweden/AI4Life_OC2_2024_11.git
cd AI4Life_OC2_2024_11
conda env create -f environment.yml
```

## Usage

Once the environment is created, you can open the prompt, activate the environment and start jupyter lab.

```
conda activate ai4life_11
jupyter lab
```

Inside the `notebooks` folder you will find notebooks for:

- [Automatic mask generation](notebooks/automatic_mask_generator.ipynb)

- [BIIGLE point to polygon generation](notebooks/BIIGLE_point_to_polygon_with_sam.ipynb)

- [Paparazzi point to point generation](notebooks/paparazzi_point_to_polygon_with_sam.ipynb)

- [Plotting and saving masks for many images](notebooks/Plot_masks_with_sam-batch.ipynb)

## Conclusion

In this tutorial, we showed how to perform semantic segmentation of objects in images with [SAM](https://github.com/facebookresearch/segment-anything).

[AI4Life](https://ai4life.eurobioimaging.eu)  is a Horizon Europe-funded project that brings together the computational and life science communities.

AI4Life has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement number 101057970. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## Development

If you wish to contribute, it would be nice to chekc the style and other things through pre-commit.
You can do this by installing pre-commit before making new commits.

```
pip install pre-commit
pre-commit install
```

## Cite

```
Author list (2024). Title. Zenodo. https://doi.org/... .
```

## License

[MIT](LICENSE)

## Contact

[SciLifeLab BioImage Informatics Facility (BIIF)](https://www.scilifelab.se/units/bioimage-informatics/)

Developed by [Mehdi Seifi](mailto:mehdi.seifi@fht.org), [Agustin Corbat and Kristina Lidayova](mailto:biif@scilifelab.se)

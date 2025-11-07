# Update - November 2, 2025:
The new dataset subsets have been uploaded.

# Update - March 22, 2025:
The dataset code and files will be updated to include the second version including 3D pose estimation data, growth monitoring and yield estimation, along with an extended detection subset of 2,000 images. Check back for updates.

# M18K: Mushroom Detection Dataset

Welcome to the Mushroom Detection Dataset repository. This dataset is a valuable resource for researchers and practitioners interested in automating agricultural processes, specifically related to mushroom harvesting, growth monitoring, and quality control. It provides a dedicated dataset for button mushroom (Agaricus bisporus) detection, comprising over 18,000 mushroom instances in 423 RGB-D image pairs.

## Overview

The dataset was collected using an Intel RealSense D405 camera, capturing realistic growth environment scenarios with comprehensive annotations. It serves as a benchmark for detection and instance segmentation algorithms in smart mushroom agriculture.

## Dataset Details

- **Mushroom Instances**: Over 18,000
- **Image Pairs**: 423 RGB-D image pairs
- **Camera Used**: Intel RealSense D405
- **Annotations**: Comprehensive annotations for detection and instance segmentation

## Download the Dataset

You can download the dataset directly from the repository using the link below:

[Download Segmentation Dataset - V1 (423 images)](https://drive.google.com/file/d/1iPANwP1k6tbz1EZRdG2qbst1tqqYsZ_x/view?usp=sharing)

[Download Segmentation Dataset - V2 (4,652 images)](https://zakerimushroom.sfo3.digitaloceanspaces.com/M18KV2.tar.zst)

[Download 3D dataset](https://zakerimushroom.sfo3.digitaloceanspaces.com/M18K_3D.zip)

[Download Growth Monitoring dataset (2,121 images)](https://zakerimushroom.sfo3.digitaloceanspaces.com/Growth.zip)

Please refer to the dataset's license file for information about usage and distribution rights.

## Environment Setup

To work with the dataset, you need to set up the environment. You can do this using the following commands:

```bash
!conda create -n m18k python=3.10 -y
!conda activate m18k
```

## Installation

To install the `m18k` package for working with the dataset, use the following command:

```bash
!pip install m18k
```

## Results and Model Zoo

Coming soon.

## References
Please cite our publication if you use our code, models, or datasets:

```bash

@Article{computers14050199,
AUTHOR = {Zakeri, Abdollah and Fawakherji, Mulham and Kang, Jiming and Koirala, Bikram and Balan, Venkatesh and Zhu, Weihang and Benhaddou, Driss and Merchant, Fatima A.},
TITLE = {M18K: A Multi-Purpose Real-World Dataset for Mushroom Detection, 3D Pose Estimation, and Growth Monitoring},
JOURNAL = {Computers},
VOLUME = {14},
YEAR = {2025},
NUMBER = {5},
ARTICLE-NUMBER = {199},
URL = {https://www.mdpi.com/2073-431X/14/5/199},
ISSN = {2073-431X},
DOI = {10.3390/computers14050199}
}
```

---

Thank you for using the Mushroom Detection Dataset. We hope this resource aids your research and development efforts in the field of smart agriculture. If you have any questions or need support, please feel free to reach out to us.

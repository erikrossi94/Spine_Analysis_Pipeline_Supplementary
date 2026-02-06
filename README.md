SPINE ANALYSIS PIPELINE - SUPPLEMENTARY MATERIAL
Dendritic spine detection and classification (mature vs immature / mushroom)
================================================================================

CONTENTS
--------
- scripts/          All Python scripts. Main entry: process_all_images.py
- models/            Trained Random Forest classifier (.pkl) and scaler
- hybrid_spine_detector.py   Core detection/classification module (place in same folder as scripts or add to PYTHONPATH)

REQUIREMENTS
------------
Python 3.8+, numpy, scipy, pandas, scikit-learn, scikit-image, tifffile, matplotlib, seaborn

USAGE (after adjusting paths inside the scripts for your system)
----------------------------------------------------------------
The pipeline was run on 100x magnification images (2x zoom). It loads the binary classification model (mushroom vs non-mushroom), processes TIF images by lineage and treatment, and outputs spine counts, densities, and percentage of mature spines per image.

Main script: scripts/process_all_images.py

REFERENCE
---------
Morphological criteria and pipeline description as in the main manuscript Methods section (Spine Analysis). Classifier: Random Forest, binary (mature/immature); cross-validated accuracy ~85.8%.

================================================================================

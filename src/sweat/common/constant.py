# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for constant management
"""

import numpy as np

FLAGS_TYPE = np.uint8
# Constant Flags
# if bit 0 activated : The pixel is invalid : input data contains nodata
MSK_INPUT_NODATA = 1 << 0
# if bit 1 activated : The pixel is invalid : input data are filtered
MSK_INPUT_FILTERED = 1 << 1
# if bit 2 activated : The pixel is invalid : input data filtered
# during some processing step
MSK_INPUT_FILTERED_DURING_PROCESSING = 1 << 2
# if bit 3 activated : The pixel is invalid : Processing failed
MSK_PROCESSING_FAILED = 1 << 3

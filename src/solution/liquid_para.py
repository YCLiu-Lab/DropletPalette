"""
Liquid phase physical property calculation module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.
"""

import numpy as np
import numba
from typing import Tuple

# define the constants related to the species
NUM_SPECIES: int = 40
COM_TOLERANCE: float = 1e-6  # species tolerance, normalization tolerance

# define the array of relative molecular weights (unit: kg/kmol)
MOLECULAR_WEIGHTS = np.array([
    # NA series (0-9)
    100.21, 114.23, 128.26, 142.29, 156.31, 170.34, 184.37, 198.39, 212.42, 226.45,  # C7H16-C16H34
    
    # IA series (10-19)
    100.21, 114.23, 128.26, 142.29, 156.31, 170.34, 184.37, 198.39, 212.42, 226.45,  # C7H16-C16H34
    
    # CA series (20-29)
    98.189, 112.22, 126.24, 140.27, 154.29, 168.32, 182.35, 196.37, 210.40, 224.43,  # C7H14-C16H32
    
    # AR series (30-39)
    92.141, 106.17, 120.19, 134.22, 148.25, 162.28, 176.3, 190.33, 204.36, 218.38   # C7H8-C16H26
])

# define the mapping of species names
SPECIES_NAMES = {
    # NA series (0-9)
    0: "NA7", 1: "NA8", 2: "NA9", 3: "NA10", 4: "NA11",
    5: "NA12", 6: "NA13", 7: "NA14", 8: "NA15", 9: "NA16",
    # IA series (10-19)
    10: "IA7", 11: "IA8", 12: "IA9", 13: "IA10", 14: "IA11",
    15: "IA12", 16: "IA13", 17: "IA14", 18: "IA15", 19: "IA16",
    # CA series (20-29)
    20: "CA7", 21: "CA8", 22: "CA9", 23: "CA10", 24: "CA11",
    25: "CA12", 26: "CA13", 27: "CA14", 28: "CA15", 29: "CA16",
    # AR series (30-39)
    30: "AR7", 31: "AR8", 32: "AR9", 33: "AR10", 34: "AR11",
    35: "AR12", 36: "AR13", 37: "AR14", 38: "AR15", 39: "AR16"
}

# define the critical temperature (unit: K)
# Aspen Plus
# AR9 725 in Aspen Plus change to 738
TC = np.array([
    540.0783, 568.7879, 594.148, 618.0478, 638.8139, 658.2474, 675.6722, 692.4759, 706.8825, 722.3919,  # NA series
    530.3689, 559.6267, 582.831, 601.6, 634, 651.1, 664, 681, 709.9, 714.9,  # IA series
    572.3138, 606.9, 630.8, 653.1, 674, 691, 707, 723, 731, 750,  # CA series
    591.89, 617.119, 638.2865, 660.4805, 675, 695, 708, 725, 738, 752  # AR series
])


# define the base density (unit: kmol/m³)
DENSITY_MOLE_BASE = np.array([
    2.334824, 2.033973, 1.824554, 1.613419, 1.470022, 1.336631, 1.168874, 1.149723, 1.0672, 0.9740954,  # NA series
    2.367927, 2.048463, 1.804949, 1.628512, 1.482161, 1.34559, 1.231408, 1.129689, 1.064955, 0.9904206,  # IA series
    2.718512, 2.246868, 2.050303, 1.785714, 1.629702, 1.475896, 1.310073, 1.219612, 1.12048, 1.064043,  # CA series
    3.155057, 2.684994, 2.268813, 2.009358, 1.672717, 1.612157, 1.472033, 1.339919, 1.212154, 1.13775  # AR series
])

# define the density calculation coefficient
DENSITY_MOLE_C1 = np.array([
    5.027197, 4.45061, 3.827231, 3.564553, 3.224435, 2.975835, 2.684114, 3.282248, 2.822846, 2.272655,  # NA series
    5.077431, 4.423531, 3.769051, 3.53043, 3.405022, 3.073679, 2.846699, 2.649641, 2.522129, 2.615479,  # IA series
    5.610533, 4.345458, 4.244367, 3.876668, 3.608319, 3.273914, 3.009745, 2.875664, 2.741187, 2.642593,  # CA series
    6.686536, 5.794586, 5.113811, 4.379605, 3.075382, 3.61023, 3.290278, 3.166587, 2.903437, 2.7663  # AR series
])

DENSITY_MOLE_C2 = np.array([
    1.523454, 1.472424, 1.78631, 1.361708, 1.406148, 1.290085, 1.908668, -0.8685738, -0.6657685, 0.9014,  # NA series
    0.9724182, 1.815531, 2.285869, 1.82782, 0.6026968, 1.023333, 1.187975, 1.300028, 0.7598754, -0.1582777,  # IA series
    1.530902, 3.999152, 1.83854, 1.873693, 0.9015797, 1.16564, 1.786576, 1.088355, 0.88976, 0.3213795,  # CA series
    1.960012, 1.735841, 1.396954, 1.577646, 6.271042, 1.454945, 1.431291, 0.7019814, 1.19842, 0.6390341  # AR series
])

DENSITY_MOLE_C3 = np.array([
    -0.6638865, -0.7348266, -1.241046, -0.5193682, -0.8540169, -0.7151879, -2.089041, 1.3621, 2.660555, -0.1956477,  # NA series
    0.659786, -1.975499, -2.136455, -1.851963, 0.8183977, -0.3641703, -1.106241, -1.584944, -0.1544191, 1.121582,  # IA series
    -0.6944685, -4.436177, -1.298671, -1.819692, 0.432799, -0.6629399, -2.367276, -0.8802128, -0.5123789, 0.2598667,  # CA series
    -0.7107034, -0.8681163, -0.4361963, -0.6188579, -9.730208, -1.441142, -1.462596, -0.03948075, -1.053466, -0.1995775  # AR series
])

DENSITY_MOLE_C4 = np.array([
    1.138402, 1.074232, 1.300843, 0.6812165, 0.9803732, 0.8404074, 1.777055, 0.1948367, -1.556255, 0.3854618,  # NA series
    0.09467573, 2.165808, 1.702389, 1.754721, -0.3502234, 0.5219873, 1.244804, 1.637858, 0.2620191, -0.2791799,  # IA series
    1.478402, 3.241813, 1.388136, 1.766646, -0.1330271, 0.9107811, 2.186752, 1.000326, 0.6378366, 0.3082864,  # CA series
    1.412205, 1.321052, 0.9533552, 0.8574952, 6.485045, 1.736824, 1.641341, 0.5384407, 1.099119, 0.6039803  # AR series
])

# define the heat of vaporization calculation coefficient
H_VAP_C1 = np.array([
    17.92212, 18.02051, 18.21311, 18.2038, 18.3103, 18.49945, 18.37194, 18.62886, 18.72621, 18.57697,  # NA series
    17.90736, 18.04, 18.12697, 18.40571, 18.32661, 17.98227, 18.05022, 18.18035, 18.55028, 18.97053,  # IA series
    17.90184, 17.91336, 18.01529, 18.11848, 18.23196, 18.39769, 18.62866, 18.45534, 18.78475, 18.6789,  # CA series
    17.94434, 18.00456, 18.1405, 18.2017, 18.36233, 18.3012, 18.34387, 18.42102, 19.09895, 18.92136   # AR series
])

H_VAP_C2 = np.array([
    1.110852, 1.029353, 1.462729, 0.9893556, 1.077554, 1.647936, 0.6302825, 1.605286, 1.6623, 0.7734324,  # NA series
    1.148012, 1.256631, 1.068637, 1.676271, 1.257359, -0.002938026, 0.1558533, 0.02177619, 1.348579, 1.526498,  # IA系列
    1.350309, 0.9208944, 0.9489737, 0.9726639, 1.08092, 1.427058, 2.007071, 1.169989, 2.229706, 1.598352,  # CA系列
    1.231299, 1.082252, 1.343098, 1.278537, 1.543233, 0.8721985, 0.9312297, 1.086461, 3.30351, 2.264817   # AR系列
])

H_VAP_C3 = np.array([
    -1.12374, -0.905851, -1.663575, -0.7646847, -0.8475598, -1.877696, 0.05941812, -1.692431, -1.683477, -0.109564,  # NA系列
    -1.108916, -1.279231, -0.9186814, -1.784176, -1.15443, 0.5468203, 0.08990575, 0.7210967, -1.288685, -0.1962959,  # IA系列
    -1.621193, -0.8534523, -0.8898833, -0.8597394, -0.958457, -1.536548, -2.462099, -1.210721, -2.961733, -1.762002,  # CA系列
    -1.35757, -1.102366, -1.496523, -1.398757, -1.783324, -0.4648225, -0.8641247, -1.106626, -4.368252, -2.498719   # AR系列
])

H_VAP_C4 = np.array([
    0.4666989, 0.3240905, 0.6943608, 0.2381593, 0.2272311, 0.7478465, -0.2430794, 0.5945089, 0.5405077, -0.2142634,  # NA系列
    0.4148294, 0.4936171, 0.3098454, 0.6151912,0.3734356, -0.1822122, 0.1236257, -0.3822444, 0.4289007, -0.8664502,  # IA系列
    0.7483666, 0.3677554, 0.3910825, 0.3214944, 0.3209138, 0.5997108, 1.00084, 0.5270122, 1.321426, 0.6677492,  # CA系列
    0.5918428, 0.4987298, 0.6247569, 0.5858075, 0.7414584, 0.02570438, 0.4084079, 0.4837281, 1.621929, 0.7575815   # AR系列
])

# define the vapor pressure calculation coefficient
VAPOR_PRESSURE_C1 = np.array([
    -7.87961, -8.089952, -8.096452, -8.581278, -8.973956, -8.746015, -10.00299, -8.888841, -9.495756, -10.64425,  # NA系列
    -7.628551, -8.031157, -8.485507, -8.964596, -9.11032, -9.716182, -9.704416, -11.00182, -9.289367, -13.73309,  # IA系列
    -7.076046, -7.868779, -7.876356, -8.046513, -8.381979, -8.492304, -8.70999, -8.887769, -9.302744, -9.213571,  # CA系列
    -7.489683, -7.540967, -7.930167, -8.177687, -7.9966, -9.346708, -8.181494, -8.784592, -9.532803, -9.640387   # AR系列
])

VAPOR_PRESSURE_C2 = np.array([
    2.1631, 2.1530, 1.5620, 2.3945, 2.6894, 1.909, 5.0499, 1.6612, 2.6635, 5.0425,  # NA系列
    1.7367, 2.1343, 2.5776, 2.7584, 3.4381, 4.3387, 3.3065, 6.1326, 2.8789, 11.8853,  # IA系列
    1.413174, 2.700335, 2.53336, 2.535799, 2.63592, 2.27583, 2.298883, 2.479826, 2.827225, 2.355331,  # CA系列
    2.0576, 1.7691, 2.2696, 2.3730, 1.3319, 4.1944, 1.3722, 1.8915, 0.3755, 2.4566   # AR系列
])

VAPOR_PRESSURE_C3 = np.array([
    -3.2332, -3.4640, -3.0560, -4.1917, -4.5175, -4.2226, -8.1389, -4.8415, -6.0870, -8.5712,  # NA系列
    -2.6282, -3.2129, -4.3531, -5.0321, -5.2909, -6.9332, -6.2131, -9.3388, -5.4972, -12.4992,  # IA系列
    -1.873893, -3.229004, -3.705645, -4.122909, -4.168205, -3.969585, -4.275862, -5.206418, -5.557091, -5.275687,  # CA系列
    -2.4878, -2.4539, -3.1220, -3.5135, -2.8188, -5.8042, -4.6191, -4.5207, -0.2573, -4.0749   # AR系列
])

VAPOR_PRESSURE_C4 = np.array([
    -3.0459, -3.5162, -4.3584, -4.2356, -4.5587, -5.1820, -3.0943, -5.2795, -5.0630, -4.1172,  # NA系列
    -3.5220, -3.5802, -3.4371, -3.9983, -3.9351, 0.0, 0.00, 0.0, -4.4136, -8.7584,  # IA系列
    -3.190502, -2.640848, -2.461743, -2.522976, -3.125167, -4.123072, -4.422117, -3.233574, -4.479429, -4.546454,  # CA系列
    -3.0796, -3.4321, -3.3032, -3.2149, -4.4511, -3.0310, -2.7783, -3.2508, -10.8871, -7.7811   # AR系列
])

VAPOR_PRESSURE_C5 = np.array([
    14.8213, 14.7265, 14.6448, 14.5640, 14.5048, 14.4062, 14.3375, 14.2521, 14.1968, 14.1860,  # NA系列
    14.8210, 14.7326, 14.6497, 14.5571, 14.5288, 14.4253, 14.3364, 14.2964, 14.1819, 14.1688,  # IA系列
    15.06289, 14.99967, 14.86304, 14.75245, 14.68876, 14.58784, 14.50112, 14.42582, 14.26409, 14.25478,  # CA系列
    15.2341, 15.0991, 14.9787, 14.8753, 14.7619, 14.6766, 14.5742, 14.4978, 14.4263, 14.3602   # AR系列
])

# define the heat capacity calculation coefficient
CP_C1 = np.array([
    354909.9, 226117.1, 424454.8, 510915.2, 621716.8, 444932.2, 531311.1, 738234, 472478, 355214,  # NA系列
    182676.5, 265066.5, 272898.4, 319960.5, 439922.5, 328307.8, 346017.7, 369094.2, 305880.6, 361362.8,  # IA系列
    134897.8, 174921.4, 193521.8, 222006.1, 196648.8, 248885.1, 275459, 269437.7, 358124.2, 500921.5,  # CA系列
    169723.4, 188594.1, 228341.6, 233522, 282734.8, 383932.6, 246801, 316938, 387527.9, 334469.4   # AR系列
])

CP_C2 = np.array([
    -1879.994, -393.2255, -2003.306, -2463.571, -3342.821, -1349.803, -2038.465, -3274.194, -892.3188, 439.1416,  # NA系列
    -282.0751, -925.7478, -652.1621, -790.0959, -1858.656, -522.7844, -433.524, -419.4126, 24.33608, -100.9787,  # IA系列
    -178.3045, -449.4858, -443.4251, -498.0671, 18.33124, -299.2721, -523.3012, 37.77046, -296.337, -1593.589,  # CA系列
    -531.5645, -618.2081, -854.6969, -581.7735, -703.4817, -1551.255, -213.7638, -412.219, -566.3627, -195.9925   # AR系列
])

CP_C3 = np.array([
    6.673906, 2.03619, 6.809594, 7.95685, 10.72092, 4.741621, 7.003119, 9.715697, 3.634134, -0.06484127,  # NA系列
    1.664878, 3.916059, 3.011432, 3.442441, 6.7641, 2.905832, 2.756195, 2.809151, 1.998308, 2.328918,  # IA系列
    1.338131, 2.435969, 2.582482, 2.813193, 1.538456, 2.513471, 3.61304, 1.775316, 2.544146, 6.082063,  # CA系列
    2.109447, 2.65454, 3.537953, 2.660975, 2.915648, 5.537016, 2.112912, 2.480698, 2.827972, 2.193985   # AR系列
])

CP_C4 = np.array([
    -0.006573532, -0.00171662, -0.006191957, -0.007059364, -0.01018517, -0.003804846, -0.00635852, -0.007943202, -0.002668192, 0.000654267,  # NA系列
    -0.001213512, -0.003752841, -0.002676023, -0.003025204, -0.006278606, -0.002464286, -0.002324392, -0.002349579, -0.001763382, -0.001998586,  # IA系列
    -0.000869601, -0.002003518, -0.002289631, -0.002422068, -0.001373525, -0.00215317, -0.003359545, -0.001543182, -0.00212153, -0.004927483,  # CA系列
    -0.001709682, -0.002488198, -0.003461254, -0.002194194, -0.002410929, -0.004801989, -0.001844438, -0.002021209, -0.002252546, -0.001810669   # AR系列
])

CP_B = np.array([
    5102.037, 4717.985, 7643.461, 9116.998, 19435.48, 6898.735, 16280.4, 12942.46, 5968.094, 2006.808,  # NA系列
    3659.773, 6671.442, 5839.452, 6387.456, 10816.3, 6398.147, 6656.283, 6884.936, 6473.766, 7040.727,  # IA系列
    3406.21, 3951.728, 6211.5, 6215.289, 5650.167, 5969.544, 8102.74, 6348.958, 6799.07, 9856.776,  # CA系列
    1892.555, 5930.565, 9816.051, 2323.261, 6289.098, 9430.189, 6437.93, 6708.888, 7296.286, 6983.237   # AR系列
])

# define the constants for the activity coefficient calculation
# UNIFAC group parameters
UNIFAC_R = np.array([0.9011, 0.6744, 0.4469, 0.2195, 0.5313, 1.2663, 1.0396])
UNIFAC_Q = np.array([0.848, 0.540, 0.228, 0.000, 0.400, 0.968, 0.660])

# UNIFAC interaction parameters
UNIFAC_A = np.array([
    [0, 0, 0, 0, 61.13, 76.50, 76.50],
    [0, 0, 0, 0, 61.13, 76.50, 76.50],
    [0, 0, 0, 0, 61.13, 76.50, 76.50],
    [0, 0, 0, 0, 61.13, 76.50, 76.50],
    [-11.12, -11.12, -11.12, -11.12, 0, 167.0, 167.0],
    [-69.70, -69.70, -69.70, -69.70, -146.8, 0, 0],
    [-69.70, -69.70, -69.70, -69.70, -146.8, 0, 0]
])

# functional group base formula
UNIFAC_FG_FORMULA = np.array([
    [2, 5, 0, 0, 0, 0, 0],  # NA系列
    [3, 3, 1, 0, 0, 0, 0],  # IA系列
    [1, 5, 1, 0, 0, 0, 0],  # CA系列
    [1, -1, 0, 0, 5, 0, 1]  # AR系列
])

# functional group increment
UNIFAC_FG_PLUS = np.array([
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0]
])

# functional group (for QCM calculation)
UNIFAC_FG = np.array([
    [2, 5, 0, 0, 0, 0, 0],  # NA系列
    [3, 3, 1, 0, 0, 0, 0],  # IA系列
    [1, 5, 1, 0, 0, 0, 0],  # CA系列
    [0, 0, 0, 0, 5, 1, 0]   # AR系列
])

# define the thermal conductivity calculation coefficient
THERMAL_CONDUCTIVITY_C1 = np.array([
    0.2518034, 0.324779, 0.1320011, 0.2054523, 0.293176, 0.2617041, 0.2729825, 0.2315637, 0.3188399, 0.124925,  # NA系列
    0.1952605, 0.2131949, 0.1830886, 0.1869343, 0.1875162, 0.1890324, 0.1951084, 0.1970643, 0.1938024, 0.2016398,  # IA系列
    0.1263551, 0.2999934, 0.2080051, 0.1786259, 0.1780748, 0.1848997, 0.1955633, 0.1949876, 0.193132, 0.1968144,  # CA系列
    0.2110555, 0.1849382, 0.1803469, 0.196229, 0.2053044, 0.2300362, 0.2002354, 0.2087614, 0.2119798, 0.2142812   # AR系列
])

THERMAL_CONDUCTIVITY_C2 = np.array([
    -0.00058022, -0.001251982, 0.00042388, -0.000253337, -0.000927561, -0.000447201, -0.000742848, -0.000400985, -0.001110158, 0.000356985,  # NA系列
    -0.000207548, -0.000397212, -0.000295845, -0.000293936, -0.000289016, -0.000279674, -0.000296627, -0.000296039, -0.000276737, -0.000295973,  # IA系列
    0.000228751, -0.001335119, -0.000432363, -0.000239416, -0.000221934, -0.000251821, -0.000279849, -0.000266521, -0.000225004, -0.000268693,  # CA系列
    -0.000230093, -8.41E-05, -8.59E-05, -0.000236193, -0.000431036, -0.000623614, -0.000231707, -0.000270114, -0.000241394, -0.000270697   # AR系列
])

THERMAL_CONDUCTIVITY_C3 = np.array([
    4.62E-07, 2.57E-06, -2.07E-06, -5.26E-08, 1.71E-06, 1.59E-08, 1.21E-06, 3.84E-07, 2.27E-06, -1.26E-06,  # NA系列
    -5.05E-07, 2.13E-07, 5.09E-07, 4.90E-07, 4.61E-07, 4.29E-07, 4.54E-07, 4.46E-07, 3.95E-07, 4.27E-07,  # IA系列
    -1.39E-06, 3.14E-06, 3.94E-07, 3.53E-07, 3.03E-07, 3.66E-07, 4.13E-07, 3.71E-07, 2.73E-07, 3.68E-07,  # CA系列
    -2.45E-07, -5.34E-07, -4.76E-07, 4.49E-08, 8.08E-07, 1.27E-06, 2.90E-07, 3.68E-07, 2.86E-07, 3.52E-07   # AR系列
])

THERMAL_CONDUCTIVITY_C4 = np.array([
    5.73E-11, -2.11E-09, 2.03E-09, 1.94E-10, -1.33E-09, 3.38E-10, -8.41E-10, -2.20E-10, -1.78E-09, 9.24E-10,  # NA系列
    8.19E-10, 0, -6.74E-10, -6.29E-10, -5.52E-10, -5.02E-10, -5.05E-10, -4.83E-10, -4.10E-10, -4.33E-10,  # IA系列
    1.43E-09, -2.68E-09, -1.05E-10, -4.43E-10, -3.77E-10, -4.10E-10, -4.51E-10, -3.95E-10, -3.26E-10, -3.60E-10,  # CA系列
    3.85E-10, 5.92E-10, 5.83E-10, 0, -7.13E-10, -1.04E-09, -3.51E-10, -3.95E-10, -3.45E-10, -3.61E-10   # AR系列
])

# define the constants for the diffusion coefficient calculation
# boiling point temperature (unit: K)
T_BOILING = np.array([
    371.533, 398.779, 423.796, 447.2649, 469.037, 489.435, 508.616, 526.689, 543.798, 560.07,  # NA系列
    363.116, 390.7767, 416.11, 440.05, 462.348, 484.318, 503.515, 521.96, 541.45, 559,  # IA系列
    374.0441, 404.916, 429.8516, 454.051, 476.657, 498.005, 518.039, 536.415, 553.45, 570.825,  # CA系列
    383.73, 409.3192, 432.3578, 456.4243, 476.32, 498.73, 515.07, 536.12, 551.85, 571.026   # AR系列
])

# viscosity calculation type
VISCOSITY_TYPE = np.array([
    1, 1, 1, 1, 1, 0, 0, 1, 1, 0,  # NA系列
    0, 1, 0, 1, 1, 0, 0, 1, 1, 0,  # IA系列
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1,  # CA系列
    0, 0, 1, 0, 1, 0, 1, 1, 1, 0   # AR系列
])

# viscosity calculation coefficient
VISCOSITY_C1 = np.array([
    1.24E-05, 1.52E-05, 2.24E-05, 1.49E-05, 2.05E-05, -14.41467, -13.51084, 2.19E-05, 1.68E-05, -13.31999,  # NA系列
    -16.31876, 1.18E-06, -15.77989, 1.73E-05, 1.71E-05, -15.96402, -15.70299, 1.64E-05, 1.38E-05, -14.34837,  # IA系列
    1.88E-05, 7.61E-05, 3.95E-05, 3.28E-05, 5.08E-05, -12.00933, 7.51E-05, 3.64E-05, 5.62E-05, 4.25E-05,  # CA系列
    -14.13487, -15.33478, 3.02E-05, -15.68655, 3.57E-05, -14.76103, 2.34E-05, 1.56E-05, 1.84E-05, -14.04636   # AR系列
])

VISCOSITY_C2 = np.array([
    2.929096, 2.774066, 2.174561, 2.853798, 2.562408, 4754.647, 3972.807, 2.584701, 2.883181, 4095.199,  # NA系列
    5565.086, 1.043895, 5436.034, 2.751356, 2.740532, 6488.879, 6319.413, 2.855019, 2.982286, 5124.207,  # IA系列
    2.138409, 0.7803176, 1.803119, 1.964264, 1.565736, 2824.19, 1.232657, 2.028965, 1.497938, 1.86889,  # CA系列
    3836.384, 5276.407, 2.032466, 5981.264, 2.05478, 5280.555, 2.550972, 2.936851, 2.620055, 4969.644   # AR系列
])

VISCOSITY_C3 = np.array([
    0.04205816, 0.01229823, 0.366305, 0.00353968, 0.155587, -1221106, -971593.1, 0.2683876, 0.1551264, -1055248,  # NA系列
    -1356671, 0.8655421, -1340191, 0.06056141, 0.1308882, -1869010, -1811549, 0.1848407, 0.0604602, -1396348,  # IA系列
    0.2880612, 0.8311653, 0.365141, 0.3111631, 0.5509251, -675101.1, 0.746973, 0.346774, 0.5153761, 0.396486,  # CA系列
    -831695.6, -1338930, 0.240996, -1621894, 0.04241543, -1412694, 0.1505853, 0.03653179, 0.6504159, -1365889   # AR系列
])

VISCOSITY_C4 = np.array([
    578.56, 601.1452, 692.8336, 655.0959, 683.0293, 148501600, 125695800, 740.7109, 771.1342, 150242600,  # NA系列
    132066500, 2685.477, 139328200, 643.3326, 680.2668, 227252700, 224220500, 745.8008, 791.788, 187627200,  # IA系列
    795.6238, 716.0694, 691.2704, 753.3374, 730.3373, 103666900, 737.3131, 809.9692, 842.47, 857.0411,  # CA系列
    83001950, 141182600, 688.1892, 183461500, 733.5471, 170812600, 753.942, 775.1901, 835.4153, 186949700   # AR系列
])

VISCOSITY_C5 = np.array([
    114.8583, 145.4281, 73.18027, 171.2555, 140.0942, 268.1, 273.15, 130.6535, 152.5446, 293.138,  # NA系列
    273.1, -402.7825, 283.15, 152.4645, 140.8033, 270, 270, 142.4098, 178.6861, 273.15,  # IA系列
    71.39954, 22.04783, 98.19912, 105.8585, 86.96575, 270, 83.62071, 129.0699, 107.9376, 127.1157,  # CA系列
    183.412, 233.1577, 114.9895, 278.097, 189.2849, 283.144, 150.2867, 184.9577, 31.38227, 273.15   # AR系列
])

# temperature range parameters
DENSITY_T_MIN = np.array([
    182.6, 216.9, 219.6, 243.5, 247.6, 263.5, 267.7, 279.6, 283.0, 291.3,  # NA系列
    154.9, 164.1, 192.8, 198.4, 224.3, 226.3, 247.1, 246.6, 264.4, 263.8,  # IA系列
    146.7, 161.8, 178.2, 198.4, 210.0, 225.6, 232.2, 273.1, 220.0, 271.4,  # CA系列
    178.1, 178.2, 171.6, 185.1, 194.7, 209.7, 220.0, 236.0, 220.0, 258.7   # AR系列
])

DENSITY_T_MAX = np.array([
    540.0, 568.7, 594.1, 618.0, 638.8, 658.2, 675.6, 692.4, 706.8, 722.3,  # NA系列
    530.3, 559.6, 582.8, 601.6, 634.0, 651.1, 664.0, 681.0, 709.9, 714.9,  # IA系列
    572.3, 606.9, 630.8, 653.1, 674.0, 691.0, 707.0, 723.0, 731.0, 750.0,  # CA系列
    591.8, 617.1, 638.2, 660.4, 675.0, 695.0, 708.0, 725.0, 722.0, 752.0   # AR系列
])

@numba.njit(cache=True)
def calculate_precompute_values(temperature: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray]:
    """precompute the common values to improve performance
    
    Args:
        temperature: temperature (K)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray]:
            - Tr: reduced temperature
            - tao: (1-Tr)
            - tao_035: (1-Tr)^0.35
            - tao_2: (1-Tr)^2
            - tao_3: (1-Tr)^3
            - T: temperature
            - T2: temperature square
            - T3: temperature cube
            - ln_1_minus_Tr: ln(1-Tr)
    """
    # calculate the temperature power (scalar operation)
    T = temperature
    T2 = T * T
    T3 = T2 * T
    
    # calculate the reduced temperature (vector operation)
    Tr = np.where(T >= TC, 0.999999, T / TC)
    tao = 1 - Tr
    tao_035 = tao ** 0.35
    tao_2 = tao * tao  # use multiplication instead of power operation
    tao_3 = tao_2 * tao
    ln_1_minus_Tr = np.log(1 - Tr)
    
    return Tr, tao, tao_035, tao_2, tao_3, T, T2, T3, ln_1_minus_Tr

@numba.njit(cache=True)
def calculate_activity_coefficients(composition: np.ndarray, temperature: float) -> np.ndarray:
    """calculate the activity coefficient matrix
    
    calculation formula:
    ln(γi) = ln(γi^C) + ln(γi^R)
    where:
    - γi^C: combination term activity coefficient
    - γi^R: residual term activity coefficient
    
    Args:
        composition: component mole fraction array, shape (40,)
        temperature: temperature (K)
        
    Returns:
        np.ndarray: activity coefficient matrix, shape (40,)
    """
    # calculate the functional group number matrix
    D_v_k = np.zeros((40, 7))
    for k in range(10):
        D_v_k[k, :] = UNIFAC_FG_FORMULA[0, :] + np.array([0, k, 0, 0, 0, 0, 0])  # NA系列
        D_v_k[k+10, :] = UNIFAC_FG_FORMULA[1, :] + np.array([0, k, 0, 0, 0, 0, 0])  # IA系列
        D_v_k[k+20, :] = UNIFAC_FG_FORMULA[2, :] + np.array([0, k, 0, 0, 0, 0, 0])  # CA系列
        D_v_k[k+30, :] = UNIFAC_FG_FORMULA[3, :] + np.array([0, k, 0, 0, 0, 0, 0])  # AR系列
    
    # special treatment for AR7
    D_v_k[30, :] = np.array([0, 0, 0, 0, 5, 1, 0])
    
    # calculate the combination term activity coefficient
    D_v_i = D_v_k @ UNIFAC_R
    D_q_i = D_v_k @ UNIFAC_Q
    D_v_i_sum = np.sum(D_v_i * composition)
    D_V_i = D_v_i / D_v_i_sum
    D_q_i_sum = np.sum(D_q_i * composition)
    D_F_i = D_q_i / D_q_i_sum
    D_VS_i = D_V_i / D_F_i
    D_VS_term = 1 - D_VS_i + np.log(D_VS_i)
    D_lnr_c1 = 1 - D_V_i + np.log(D_V_i)
    D_lnr_c2 = -5 * D_q_i * D_VS_term
    D_lnr_c = D_lnr_c1 + D_lnr_c2
    D_r_c = np.exp(D_lnr_c)
    
    # calculate the residual term activity coefficient
    phi = np.exp(-1 * UNIFAC_A / temperature)
    D_X_m_i = D_v_k / np.sum(D_v_k, axis=1)[:, np.newaxis]
    D_X_m_mean = np.sum(D_v_k * composition.reshape((40, 1)), axis=0) / np.sum(D_v_k * composition.reshape((40, 1)))
    D_theta_mean = (UNIFAC_Q * D_X_m_mean) / np.sum(UNIFAC_Q * D_X_m_mean)
    D_theta_i = (UNIFAC_Q * D_X_m_i) / np.sum(UNIFAC_Q * D_X_m_i, axis=1)[:, np.newaxis]
    
    # calculate the complex term
    D_complex_term = np.zeros(7)
    D_complex_term_i = np.zeros((40, 7))
    
    # use explicit loop to calculate the complex term
    for k in range(7):
        sum_up = D_theta_mean * phi[k, :]
        sum_down = np.sum(D_theta_mean * phi, axis=1)
        D_complex_term[k] = (1 - np.log(np.sum(D_theta_mean * phi[:, k])) - np.sum(sum_up / sum_down))
        
        for i in range(40):
            sum_up_i = D_theta_i[i, :] * phi[k, :]
            sum_down_i = np.sum(D_theta_i[i, :] * phi, axis=1)
            D_complex_term_i[i, k] = (1 - np.log(np.sum(D_theta_i[i, :] * phi[:, k])) - np.sum(sum_up_i / sum_down_i))
    
    D_d_complex_term = D_complex_term - D_complex_term_i
    D_d_ln = UNIFAC_Q * D_d_complex_term
    D_lnr_r = np.sum(D_v_k * D_d_ln, axis=1)
    D_r_r = np.exp(D_lnr_r)
    
    # calculate the final activity coefficient matrix
    activity_matrix = D_r_r * D_r_c
    
    return activity_matrix

@numba.njit(cache=True)
def calculate_density_mole(tao: np.ndarray, tao_035: np.ndarray,
                         tao_2: np.ndarray, tao_3: np.ndarray) -> np.ndarray:
    """计算摩尔密度矩阵
    
    计算公式:
    rho = rho_base + C1*(1-Tr)^0.35 + C2*(1-Tr) + C3*(1-Tr)^2 + C4*(1-Tr)^3
    
    其中:
    - rho_base: 基准密度
    - C1-C4: 密度计算系数
    - Tr: 对比温度 (T/Tc)
    
    参数:
        tao: (1-Tr)矩阵
        tao_035: (1-Tr)^0.35矩阵
        tao_2: (1-Tr)^2矩阵
        tao_3: (1-Tr)^3矩阵
        
    返回:
        np.ndarray: 摩尔密度矩阵，形状为(40,)
    """
    # 计算密度矩阵
    density_matrix = (DENSITY_MOLE_BASE + 
                     DENSITY_MOLE_C1 * tao_035 + 
                     DENSITY_MOLE_C2 * tao + 
                     DENSITY_MOLE_C3 * tao_2 + 
                     DENSITY_MOLE_C4 * tao_3)
    density_matrix[34] = 1/2 * (density_matrix[33] * 134.22 + density_matrix[35] * 162.28)/148.25
    return density_matrix

@numba.njit(cache=True)
def calculate_cp_mole(tao: np.ndarray, T: float, T2: float, T3: float) -> np.ndarray:
    """计算摩尔比热容矩阵
    
    计算公式:
    cp = B/(1-Tr) + C1 + C2*T + C3*T^2 + C4*T^3
    
    其中:
    - B: 基准比热容
    - C1-C4: 比热容计算系数
    - Tr: 对比温度 (T/Tc)
    
    参数:
        tao: (1-Tr)矩阵
        T: 温度 (K)
        T2: 温度平方
        T3: 温度立方
        
    返回:
        np.ndarray: 摩尔比热容矩阵，形状为(40,)
    """
    # 计算比热容矩阵
    cp_matrix = (CP_B / tao +
                CP_C1 + 
                CP_C2 * T + 
                CP_C3 * T2 + 
                CP_C4 * T3)
    cp_matrix[4] = 1/2 * (cp_matrix[3]*142.29 + cp_matrix[5]*170.34)/156.31
    cp_matrix[6] = 1/2 * (cp_matrix[5]*170.3  + cp_matrix[7]* 198.39)/184.37
    cp_matrix[8] = 1/2 * (cp_matrix[7]*198.39 + cp_matrix[9]*226.45)/212.42
    cp_matrix[38] = 1/2 * (cp_matrix[37]*190.33 + cp_matrix[39]*218.38) /204.36
    cp_matrix[10:20] = cp_matrix[0:10]
    return cp_matrix

@numba.njit(cache=True)
def calculate_heat_vaporization_mole(Tr: np.ndarray, ln_1_minus_Tr: np.ndarray) -> np.ndarray:
    """计算摩尔汽化热矩阵
    
    计算公式:
    H = exp(C1 + C2*ln(1-Tr) + C3*Tr*ln(1-Tr) + C4*Tr^2*ln(1-Tr))
    
    其中:
    - C1-C4: 汽化热计算系数
    - Tr: 对比温度 (T/Tc)
    
    参数:
        Tr: 对比温度矩阵
        ln_1_minus_Tr: ln(1-Tr)矩阵
        
    返回:
        np.ndarray: 摩尔汽化热矩阵，形状为(40,)
    """
    # 计算汽化热矩阵
    heat_matrix = np.zeros((40, ), dtype=np.float64)  # 明确指定维度和类型
    heat_matrix = np.exp(
        H_VAP_C1 + 
        H_VAP_C2 * ln_1_minus_Tr + 
        H_VAP_C3 * Tr * ln_1_minus_Tr + 
        H_VAP_C4 * Tr * Tr * ln_1_minus_Tr
    )
    heat_matrix[5] = 1/2 * (heat_matrix[4] + heat_matrix[6])
    heat_matrix[7] = 1/2 * (heat_matrix[6] + heat_matrix[8])
    heat_matrix[10:20] = heat_matrix[0:10]
    heat_matrix[20:30] = heat_matrix[0:10]
    heat_matrix[30:40] = heat_matrix[0:10]
    return heat_matrix

@numba.njit(cache=True)
def calculate_thermal_conductivity(T: float, T2: float, T3: float) -> np.ndarray:
    """计算导热系数矩阵
    
    计算公式:
    k = C1 + C2*T + C3*T^2 + C4*T^3
    
    其中:
    - C1-C4: 导热系数计算系数
    - T: 温度
    
    参数:
        T: 温度 (K)
        T2: 温度平方
        T3: 温度立方
        
    返回:
        np.ndarray: 导热系数矩阵，形状为(40,)
    """
    # 计算导热系数矩阵
    k_matrix = np.zeros((40, ), dtype=np.float64)
    k_matrix = (THERMAL_CONDUCTIVITY_C1 + 
               THERMAL_CONDUCTIVITY_C2 * T + 
               THERMAL_CONDUCTIVITY_C3 * T2 + 
               THERMAL_CONDUCTIVITY_C4 * T3)
    k_matrix[2] = 1/2 * (k_matrix[1] + k_matrix[3])
    k_matrix[5] = 1/2 * (k_matrix[4] + k_matrix[6])
    k_matrix[8] = 1/2 * (k_matrix[7] + k_matrix[9])
    k_matrix[10:20] = k_matrix[0:10]
    k_matrix[20:30] = k_matrix[0:10]
    k_matrix[32] = 1/2 * (k_matrix[31] + k_matrix[33])
    k_matrix[34] = 1/2 * (k_matrix[33] + k_matrix[35])
    return k_matrix

@numba.njit(cache=True)
def calculate_viscosity(T: float, T2: float, T3: float) -> np.ndarray:
    """计算粘度矩阵
    
    计算公式:
    对于TYPE=0:
        mu = exp(C1 + C2/T + C3/T^2 + C4/T^3)
    对于TYPE=1:
        y = (C4-C5)/(T-C5) - 1
        mu = C1*exp((C2 + C3*y)*(y^(1/3)))
    
    其中:
    - C1-C5: 粘度计算系数
    - T: 温度
    
    参数:
        T: 温度 (K)
        T2: 温度平方
        T3: 温度立方
        
    返回:
        np.ndarray: 粘度矩阵，形状为(40,)
    """
    # 计算TYPE=0的粘度
    mu_type0 = np.exp(VISCOSITY_C1 + VISCOSITY_C2/T + VISCOSITY_C3/T2 + VISCOSITY_C4/T3)
    
    # 计算TYPE=1的粘度
    y = (VISCOSITY_C4 - VISCOSITY_C5) / (T - VISCOSITY_C5) - 1
    y = np.where(y < 0, 0, y)
    mu_type1 = VISCOSITY_C1 * np.exp((VISCOSITY_C2 + VISCOSITY_C3 * y) * (y ** (1/3)))
    
    # 根据TYPE选择对应的粘度
    mu_matrix = np.where(VISCOSITY_TYPE == 1, mu_type1, mu_type0)
    mu_matrix[2] = 1/2 * (mu_matrix[1] + mu_matrix[3])
    mu_matrix[10:20] = mu_matrix[0:10]
    mu_matrix[21] = 1/2 * (mu_matrix[20] + mu_matrix[22])
    mu_matrix[32] = 1/2 * (mu_matrix[31] + mu_matrix[33])
    mu_matrix[34] = 1/2 * (mu_matrix[33] + mu_matrix[35])
    mu_matrix[36] = 1/2 * (mu_matrix[35] + mu_matrix[37])
    # 确保粘度不为负值
    return np.maximum(mu_matrix, 0)

@numba.njit(cache=True)
def calculate_viscosity_mean(composition: np.ndarray, viscosity_ij: np.ndarray) -> float:
    """计算混合物的粘度
    
    计算公式:
    ln(μ_mix) = Σ(xi * ln(μi))
    
    其中:
    - xi: 组分i的摩尔分数
    - μi: 组分i的粘度
    
    参数:
        composition: 组分摩尔分数数组，形状为(40,)
        viscosity_ij: 单个组分的粘度矩阵，形状为(40,)
        
    返回:
        float: 混合物的粘度
    """
    # 使用向量化操作计算混合物粘度
    ln_viscosity = np.sum(composition * np.log(viscosity_ij))
    total_moles = np.sum(composition)
    
    return np.exp(ln_viscosity / total_moles) if total_moles > 0 else 0.0

@numba.njit(cache=True)
def calculate_molecular_weight_mean(composition: np.ndarray) -> float:
    """计算混合物的平均分子量
    
    计算公式:
    M_mean = Σ(xi * Mi)
    
    其中:
    - xi: 组分i的摩尔分数
    - Mi: 组分i的分子量
    
    参数:
        composition: 组分摩尔分数数组，形状为(40,)
        
    返回:
        float: 平均分子量
    """
    return np.sum(composition * MOLECULAR_WEIGHTS)

@numba.njit(cache=True)
def calculate_density_mole_mean(composition: np.ndarray, density_mole_ij: np.ndarray) -> float:
    """计算混合物的摩尔密度
    
    计算公式:
    rho_mix = Σ(xi * ρi)
    
    其中:
    - xi: 组分i的摩尔分数
    - ρi: 组分i的摩尔密度
    
    参数:
        composition: 组分摩尔分数数组，形状为(40,)
        density_mole_ij: 单个组分的摩尔密度矩阵，形状为(40,)
        
    返回:
        float: 混合物的摩尔密度
    """
    return np.sum(composition * density_mole_ij)

@numba.njit(cache=True)
def calculate_cp_mole_mean(composition: np.ndarray, cp_mole_ij: np.ndarray) -> float:
    """计算混合物的摩尔热容
    
    计算公式:
    cp_mix = Σ(xi * cpi)
    
    其中:
    - xi: 组分i的摩尔分数
    - cpi: 组分i的摩尔热容
    
    参数:
        composition: 组分摩尔分数数组，形状为(40,)
        cp_mole_ij: 单个组分的摩尔热容矩阵，形状为(40,)
        
    返回:
        float: 混合物的摩尔热容
    """
    return np.sum(composition * cp_mole_ij)

@numba.njit(cache=True)
def calculate_heat_vaporization_mole_mean(composition: np.ndarray, heat_vaporization_mole_ij: np.ndarray) -> float:
    """计算混合物的摩尔蒸发热
    
    计算公式:
    hv_mix = Σ(xi * hvi)
    
    其中:
    - xi: 组分i的摩尔分数
    - hvi: 组分i的摩尔蒸发热
    
    参数:
        composition: 组分摩尔分数数组，形状为(40,)
        heat_vaporization_mole_ij: 单个组分的摩尔蒸发热矩阵，形状为(40,)
        
    返回:
        float: 混合物的摩尔蒸发热
    """
    return np.sum(composition * heat_vaporization_mole_ij)

@numba.njit(cache=True)
def calculate_thermal_conductivity_mean(composition: np.ndarray, thermal_conductivity_ij: np.ndarray, mass_fraction: np.ndarray) -> float:
    """计算混合物的导热系数
    
    计算公式:
    k_mix = (Σ(wi/ki^2))^(-0.5)
    
    其中:
    - wi: 组分i的质量分数
    - ki: 组分i的导热系数
    
    参数:
        composition: 组分摩尔分数数组，形状为(40,)
        thermal_conductivity_ij: 单个组分的导热系数矩阵，形状为(40,)
        mass_fraction: 质量分数数组，形状为(40,)
        
    返回:
        float: 混合物的导热系数
    """
    return (np.sum(mass_fraction * (thermal_conductivity_ij ** (-2)))) ** (-0.5)

@numba.njit(cache=True)
def calculate_vapor_pressure_mean(composition: np.ndarray, vapor_pressure_ij: np.ndarray) -> float:
    """计算混合物的饱和蒸气压
    
    计算公式:
    P_mix = Σ(xi * Pi)
    
    其中:
    - xi: 组分i的摩尔分数
    - Pi: 组分i的饱和蒸气压
    
    参数:
        composition: 组分摩尔分数数组，形状为(40,)
        vapor_pressure_ij: 单个组分的饱和蒸气压矩阵，形状为(40,)
        
    返回:
        float: 混合物的饱和蒸气压
    """
    return np.sum(composition * vapor_pressure_ij)

@numba.njit(cache=True)
def calculate_vapor_pressure(Tr: np.ndarray, tao: np.ndarray) -> np.ndarray:
    """计算组分的饱和蒸气压
    
    计算公式:
    P_sat = exp(C5 + (C1*(1-Tr) + C2*(1-Tr)^1.5 + C3*(1-Tr)^2.5 + C4*(1-Tr)^5) / Tr)
    
    其中:
    - P_sat: 饱和蒸气压 (Pa)
    - Tr: 对比温度
    - C1-C5: 饱和蒸气压计算系数
    
    参数:
        Tr: 对比温度
        tao: (1-Tr)
        
    返回:
        np.ndarray: 蒸气压矩阵，形状为(40,)
    """
    # 计算需要的tao幂次
    tao_15 = tao ** 1.5
    tao_25 = tao ** 2.5
    tao_5 = tao ** 5
    
    # 计算蒸气压矩阵
    vapor_pressure = np.exp(
        VAPOR_PRESSURE_C5 + 
        (VAPOR_PRESSURE_C1 * tao + 
         VAPOR_PRESSURE_C2 * tao_15 + 
         VAPOR_PRESSURE_C3 * tao_25 + 
         VAPOR_PRESSURE_C4 * tao_5) / Tr
    )
    
    return vapor_pressure

@numba.njit(cache=True)
def calculate_diffusion_ij(temperature: float, tao: np.ndarray, tao_035: np.ndarray,
                         tao_2: np.ndarray, tao_3: np.ndarray, T2: float, T3: float) -> np.ndarray:
    """计算单个组分的扩散系数矩阵
    
    计算公式:
    D_ij = 8.93e-8 * (Vj^0.267 / Vi^0.433) * T / μj * 1e-4
    
    其中:
    - D_ij: 组分i和j之间的扩散系数
    - Vi, Vj: 组分i和j的摩尔体积
    - T: 温度
    - μj: 组分j的粘度
    
    参数:
        temperature: 温度 (K)
        tao: (1-Tr)矩阵
        tao_035: (1-Tr)^0.35矩阵
        tao_2: (1-Tr)^2矩阵
        tao_3: (1-Tr)^3矩阵
        T2: 温度平方
        T3: 温度立方
        
    返回:
        np.ndarray: 扩散系数矩阵，形状为(4,10,4,10)
    """
    # 计算粘度
    v_l = calculate_viscosity(temperature, T2, T3)
    v_l_cp = v_l * 1e3  # 转换为cp单位
    
    # 计算摩尔体积
    V_bp = np.array([161.82, 184.17, 206.64, 229.23, 251.92, 274.71, 297.58, 320.54, 343.57, 366.67,  # NA系列
    161.38, 183.73, 206.20, 228.79, 251.48, 274.26, 297.13, 320.08, 343.11, 366.21,  # IA系列
    150.99, 173.28, 195.70, 218.23, 240.88, 263.62, 286.45, 309.36, 332.36, 355.42,  # CA系列
    122.99, 145.08, 167.33, 189.71, 212.22, 234.84, 257.55, 280.36, 303.25, 326.23  # AR系列
    ])

    # 使用广播机制计算扩散系数
    # 扩展V_bp的维度以进行广播
    V_bp_i1j1 = V_bp.reshape(4, 10, 1, 1)  # 形状变为(4,10,1,1)
    V_bp_i2j2 = V_bp.reshape(1, 1, 4, 10)  # 形状变为(1,1,4,10)
    v_l_cp_i2j2 = v_l_cp.reshape(1, 1, 4, 10)  # 形状变为(1,1,4,10)
    
    # 计算扩散系数矩阵
    D_Aj_l0 = (8.93 * 1e-8 * (V_bp_i2j2 ** 0.267) / (V_bp_i1j1 ** 0.433)) \
              * temperature / v_l_cp_i2j2 * 1e-4
    
    return D_Aj_l0

@numba.njit(cache=True)
def calculate_diffusion_mean(composition: np.ndarray, temperature: float,
                           Tr: np.ndarray, tao: np.ndarray, tao_035: np.ndarray,
                           tao_2: np.ndarray, tao_3: np.ndarray, T2: float, T3: float) -> np.ndarray:
    """计算平均扩散系数
    
    计算公式:
    D_i = (D_i_L_im^(1-xi) * D_i_L_mi^xi)
    
    其中:
    - D_i: 组分i的平均扩散系数
    - D_i_L_im: 组分i在混合物中的扩散系数
    - D_i_L_mi: 组分i在纯组分中的扩散系数
    - xi: 组分i的摩尔分数
    
    参数:
        composition: 组分摩尔分数数组，形状为(40,)
        temperature: 温度 (K)
        Tr: 对比温度
        tao: (1-Tr)矩阵
        tao_035: (1-Tr)^0.35矩阵
        tao_2: (1-Tr)^2矩阵
        tao_3: (1-Tr)^3矩阵
        T2: 温度平方
        T3: 温度立方
        
    返回:
        np.ndarray: 平均扩散系数矩阵，形状为(40,)
    """
    # 计算粘度
    v_l = calculate_viscosity(temperature, T2, T3)
    v_l_m = calculate_viscosity_mean(composition, v_l)  # 计算混合物粘度
    
    # 计算摩尔体积
    rhon_bp = calculate_density_mole(tao, tao_035, tao_2, tao_3)
    V_bp = (1 / rhon_bp) * 1e3  # cm3/mol
    
    # 计算平均摩尔体积
    V_bp_m = np.sum(V_bp * composition)
    
    # 计算扩散系数矩阵
    D_Aj_l0 = calculate_diffusion_ij(temperature, tao, tao_035, tao_2, tao_3, T2, T3)
    
    # 计算D_i_L_im
    v_l_08 = v_l ** 0.8
    D_i_L_im = np.zeros(40)
    
    # 使用向量化操作计算D_i_L_im
    for i in range(4):
        for j in range(10):
            idx = i * 10 + j
            sum_term = 0.0
            # 计算其他组分的贡献
            for i2 in range(4):
                for j2 in range(10):
                    if i2 != i or j2 != j:
                        idx2 = i2 * 10 + j2
                        sum_term += composition[idx2] * D_Aj_l0[i,j,i2,j2] * v_l_08[idx2]
            D_i_L_im[idx] = sum_term
    
    D_i_L_im = D_i_L_im / (v_l_m ** 0.8)
    
    # 计算D_i_L_mi
    D_i_L_mi = (8.93 * 1e-8 * (V_bp ** 0.267) / (V_bp_m ** 0.433)) \
               * temperature / v_l * 1e-4
    
    return D_i_L_im ** (1 - composition) * D_i_L_mi ** composition


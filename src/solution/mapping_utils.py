"""
mapping utils module

Copyright (c) 2024 Droplet Combustion Simulation 1D
Licensed under the MIT License - see LICENSE file for details.


This module provides the mapping relationship between liquid and gas species, used to establish the corresponding relationship between the indices of multi-component fuel in liquid and gas phases, please ensure that the arrangement order of species in gas phase mechanism is consistent with that in liquid phase.

"""
# 液相组分名称字典
LIQUID_SPECIES_NAMES = {
    # NA系列 (0-9)
    0: "NA7", 1: "NA8", 2: "NA9", 3: "NA10", 4: "NA11",
    5: "NA12", 6: "NA13", 7: "NA14", 8: "NA15", 9: "NA16",
    # IA系列 (10-19)
    10: "IA7", 11: "IA8", 12: "IA9", 13: "IA10", 14: "IA11",
    15: "IA12", 16: "IA13", 17: "IA14", 18: "IA15", 19: "IA16",
    # CA系列 (20-29)
    20: "CA7", 21: "CA8", 22: "CA9", 23: "CA10", 24: "CA11",
    25: "CA12", 26: "CA13", 27: "CA14", 28: "CA15", 29: "CA16",
    # AR系列 (30-39)
    30: "AR7", 31: "AR8", 32: "AR9", 33: "AR10", 34: "AR11",
    35: "AR12", 36: "AR13", 37: "AR14", 38: "AR15", 39: "AR16"
}

# 气相组分名称字典
GAS_liquid_dict = {
    # NA系列对应的NC系列
    "NA7": "NC7H16", "NA8": "NC8H18", "NA9": "NC9H20", "NA10": "NC10H22", "NA11": "NC11H24",
    "NA12": "NC12H26", "NA13": "NC13H28", "NA14": "NC14H30", "NA15": "NC15H32", "NA16": "NC16H34",
    
    # IA系列对应的C系列-2
    "IA7": "C7H16-2", "IA8": "C8H18-2", "IA9": "C9H20-2", "IA10": "C10H22-2", "IA11": "C11H24-2",
    "IA12": "C12H26-2", "IA13": "C13H28-2", "IA14": "C14H30-2", "IA15": "C15H32-2", "IA16": "C16H34-2",
    
    # CA系列对应的CH3cC6H11系列
    "CA7": "CH3cC6H11",  "CA8": "C2H5cC6H11", "CA9": "C3H7cC6H11", "CA10": "C4H9cC6H11", "CA11": "C5H11cC6H11",
    
    # AR系列对应关系
    "AR7": "C6H5CH3",  "AR8": "A1C2H5", "AR9": "A1C3H7", "AR10": "A1C4H9", "AR11": "A1C5H11", "AR12": "A1C6H13"
}

def init_species_mapping(gas_species_names: list) -> dict:
    """
    初始化液相和气相组分的对应关系
    
    Args:
        gas_species_names: 气相组分名称列表
        
    Returns:
        dict: 组分对应关系字典，键为液相组分索引，值为气相组分索引
    """
    # 建立组分对应关系
    species_mapping = {}
    print("\n=== the liquid and gas species mapping is as follows: ===")
    # 遍历液相组分名称
    for liquid_idx, liquid_name in LIQUID_SPECIES_NAMES.items():
        # 获取对应的气相组分名称
        gas_name = GAS_liquid_dict.get(liquid_name)
        
        # 如果找到了对应的气相组分名称，并且该组分存在于气相组分列表中
        if gas_name and gas_name in gas_species_names:
            species_mapping[liquid_idx] = gas_species_names.index(gas_name)
            print(f"    liquid phase {liquid_name} corresponds to gas phase {gas_name}")
        else:
            print(f"    warning: liquid phase {liquid_name} does not find the corresponding gas phase, mapped to None")
            species_mapping[liquid_idx] = None
    print("="*50+"\n")
    return species_mapping

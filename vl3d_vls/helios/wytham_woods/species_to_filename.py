
"""
What this files does
"""

from pathlib import Path
import re
import os
import json

natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', str(s))]

if __name__ == '__main__':

    # inputs
    tree_files = Path(r"I:\Wytham_Woods\tree_research-wytham_woods_3d_model-7bee82c68924\leaves").glob("*.obj")
    helios_folder = r"H:\helios\data\sceneparts\wytham_woods"
    species_look_up_file = r"I:\Wytham_Woods\tree_research-wytham_woods_3d_model-7bee82c68924\DART_models\3D-explicit model\Trees\tree_has_leaf\species.json"

    # read all files to list and sort (natural sort)
    trees_list = list(tree_files)
    trees_list_sorted = sorted(trees_list, key=natsort)

    # open files and look up species ID - write tree ID and species ID into dictionary and rename files in HELIOS directory
    trees_dict = {}
    for i, file in enumerate(trees_list_sorted):
        search_str = "usemtl"
        with open(file) as f:
            for line in f:
                if line.startswith(search_str):
                    species_id = line.split(" ")[1].strip().split("_")[1]
                    break
        tree_id = f"tree_{i+1}"
        trees_dict[tree_id] = species_id
        old_name = str(Path(helios_folder) / f"{tree_id}.obj")
        new_name = str(Path(helios_folder) / f"{tree_id}_{species_id}.obj")
        # os.rename(old_name, new_name)

    with open(species_look_up_file, "w") as f:
        f.write(json.dumps(trees_dict, indent=4))

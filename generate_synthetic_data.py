import numpy as np
from PIL import Image, ImageDraw
import os
from tqdm.auto import tqdm
import json

from diversify_samples import apply_augmentations
from helpers import labels_df, items_with_tags, apply_random_offsets, tag_dict, fit_text_in_box, validate_bb

raw_bbs = labels_df[["bbox_x", "bbox_y", "bbox_width", "bbox_height"]].to_numpy()
bbs = [validate_bb(bb) for bb in raw_bbs]
NUM_RECEIPTS_TO_GENERATE = 100
font_paths = os.listdir("fonts")
data = []

for _ in tqdm(range(NUM_RECEIPTS_TO_GENERATE), total=NUM_RECEIPTS_TO_GENERATE):
    image = Image.open("2.jpg")
    font_path = "D:/Personal/Uni/4_point/thesis/projects/trainSyntheticDataset/fonts/" + font_paths[np.random.randint(0, len(font_paths))]
    draw = ImageDraw.Draw(image)

    strings_used = []
    modified_bbs = []
    for idx, bb in enumerate(bbs[:len(items_with_tags)]):
        # Apply random offsets
        max_horizontal_offset = 10  # Adjust as needed
        max_vertical_offset = 5    # Adjust as needed
        new_bb = apply_random_offsets(bb, max_horizontal_offset, max_vertical_offset, modified_bbs)
        modified_bbs.append(new_bb)

        string = items_with_tags[idx]
        if string.startswith("#") and string.endswith("#"):
            string = tag_dict[string]()
        fit_text_in_box(image, string, new_bb, font_path, 10, show_BB=False)
        strings_used.append(string)

    image = apply_augmentations(image)
    num_images = len(os.listdir("D:/Personal/Uni/4_point/thesis/samples/synthetic_dataset/generated"))
    image.save(f"D:/Personal/Uni/4_point/thesis/samples/synthetic_dataset/generated/{num_images + 1}.jpg")

    modified_bbs = [validate_bb(bb) for bb in modified_bbs]
    strings_and_bbs = [] #strings and bbs for a single image
    for string, (bb) in zip(strings_used, modified_bbs):
        strings_and_bbs.append([string, bb[0], bb[1], bb[2], bb[3]])
    data.append({
            "image_path": f"D:/Personal/Uni/4_point/thesis/samples/synthetic_dataset/{num_images + 1}.jpg",
            "strings_and_bbs": strings_and_bbs
        })


#convert to correct number type so we can store as json
def default_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError


#store labels in a json file
with open(f"D:/Personal/Uni/4_point/thesis/samples/synthetic_dataset/labels.json", "w") as f:
    json.dump(data, f, indent=4, default=default_converter)

import json
import os


def load_dataset(train_split=0.8, folder_path="CRAFT_data/"):
    # first train
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        print("FOLDER ALREADY EXISTS, WILL NOT OVERWRITE SO RETURNING. REMOVE FOLDER IF YOU WANT TO RECREATE")
        return

    # TODO open labels.json file
    path_to_labels_file = "D:/Personal/Uni/4_point/thesis/samples/synthetic_dataset/labels.json"
    with open(path_to_labels_file) as f:
        labels = json.load(f)

    full_folder_path = folder_path + "ch4_training_localization_transcription_gt"
    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)
    split_idx = int(len(labels) * train_split)

    for idx, (image_info) in enumerate(labels):
        if idx == split_idx:
            full_folder_path = folder_path + "ch4_test_localization_transcription_gt"
            if not os.path.exists(full_folder_path):
                os.makedirs(full_folder_path)

        image_path, labels_and_bbs = image_info["image_path"], image_info["strings_and_bbs"]
        image_name = image_path.split('/')[-1].split('.')[0]
        # now take all labels and bounding boxes and convert to correct format
        with open(f'{full_folder_path}/gt_{image_name}.txt', 'w', encoding='utf-8') as f:
            for (lbl, x1, y1, w,
                 h) in labels_and_bbs:  # x1,y1 is top left corner (2 is top right, 3 is bot right, 4 is bot left)
                x2 = x1 + w
                y2 = y1
                x3 = x1 + w
                y3 = y1 + h
                x4 = x1
                y4 = y1 + h
                # full_label_string = [x1, y1, x2, y2, x3, y3, x4, y4, lbl]
                full_label_string = f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{lbl}"
                # then save to txt
                f.write(full_label_string + '\n')
            f.close()

load_dataset()


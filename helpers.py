import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm.auto import tqdm
import json
import logging

# Configure logging
logging.basicConfig(
    filename='dataset_generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Read labels
labels_df = pd.read_csv("labels_annotation1.csv", sep=",")
bbs = labels_df[["bbox_x", "bbox_y", "bbox_width", "bbox_height"]].to_numpy()

# List of items with tags
items_with_tags = [
    "ТОВАРИСТВО", "З", "ОБМЕЖЕНОЮ", "ВІДПОВІДАЛЬНІСТЮ",
    "ЄВРОТЕКС ТРЕЙД", "МАГАЗИН",
    "М. КИЇВ,", "ШЕВЧЕНКІВСЬКИЙ", "РАЙОН,",
    "ВУЛИЦЯ", "ВЕЛИКА", "ВАСИЛЬКІВСЬКА", "БУД.", "ПТП", "Б,",
    "Г В", "ОСЯХ", "1-84,", "ЛІТЕРА", "А",
    "ПН", "#12number#",
    "КАСИР:", "ПОТОРОЧА", "О.Д.",
    "УКТЗЕД", "#10number#",
    "ШТРИХКОД", "#13number#",
    "#item#", "ЧОЛ.", "COMFORT", "(КАШЕМИР)", "15",
    "#comma_number#", "A", "с-66сп,", "Р.29,", "000", "ТЕМНО-СІРИЙ",
    "ЗНИЖКА", "-7,00", "А",
    "УКТЗЕД", "#10number#",
    "ШТРИХКОД", "#13number#",
    "#item#", "ЧОЛ.", "Л&П", "240", "(МАХРОВІ),",
    "Р", "#comma_number#", "А", ".44-45,", "00", "ТЕМНО_СІРИЙ",
    "ЗНИЖКА", "-4,00", "А",
    "УКТЗЕД", "#10number#",
    "ШТРИХКОД", "#13number#",
    "ПАКЕТ", "МАЛЕНЬКИЙ", "#comma_number#", "А",
    "РАЙФАЙЗЕН БАНК",
    "ІДЕНТ. ЕКВАЙРА", "6211074",
    "ТЕРМІНАЛ", "КОМІСІЯ", "F0137562",
    "#comma_number#", "ГРН",
    "ОПЛАТА",
    "ЕПЗ", "5167********7340",
    "ПЛАТІЖНА", "СИСТЕМА", "MASTERCARD",
    "КОД", "АВТ.", "816360",
    "RRN", "#12number#",
    "КАСИР",
    "ДЕРЖАТЕЛЬ", "ЕПЗ",
    "КАРТКА", "#comma_number#", "ГРН",
    "СУМА", "#comma_number#",
    "ПДВ_А =", "20,00%", "#comma_number#",
    "ДО СПЛАТИ", "#comma_number#",
    "#!!!УВАГА!!!",
    "#ПАНЧІШНО-ШКАРПЕТКОВІ",
    "#ВИРОБИ І НАТІЛЬНА",
    "#БІЛИЗНА ОБМІНУ", "ТА",
    "#ПОВЕРНЕННЮ", "НЕ", "ПІДЛЯГАЮТЬ",
    "ФН", "ЧЕКУ", "V-qbdzy8fu8",
    "#date#", "#time#",
    "РЕЖИМ:", "ОНЛАЙН",
    "ФН", "ПРО", "#10number#",
    "ФІСКАЛЬНИЙ", "ЧЕК",
    "ВЧАСНО-КАСА"
]


# Random data generation functions
def get_random_time():
    """Return random time in format: hh:mm:ss"""
    hour = f"{np.random.randint(0, 24):02d}"
    minute = f"{np.random.randint(0, 60):02d}"
    second = f"{np.random.randint(0, 60):02d}"
    return f"{hour}:{minute}:{second}"


def get_random_date():
    """Return random date in format: dd.mm.yy"""
    day = f"{np.random.randint(1, 32):02d}"  # Days 01-31
    month = f"{np.random.randint(1, 13):02d}"  # Months 01-12
    year = f"{np.random.randint(0, 100):02d}"  # Years 00-99
    return f"{day}.{month}.{year}"


def validate_all_possible_items(file_path):
    """Validate the all_possible_items.txt file."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove empty lines and strip whitespace
    valid_lines = [line.strip() for line in lines if line.strip()]

    if not valid_lines:
        raise ValueError("all_possible_items.txt is empty or contains only whitespace.")

    return valid_lines


def get_random_item(all_possible_items):
    """Return a random item from the pre-validated list."""
    return np.random.choice(all_possible_items)


def get_random_numbers(length):
    """Return a string of random digits of specified length."""
    return ''.join(str(np.random.randint(0, 10)) for _ in range(length))


def get_random_comma_number():
    """Return a random number with a comma, e.g., '123,45'."""
    if np.random.rand() < 0.1:
        # 10% chance to have 3 digits before comma
        return f"{np.random.randint(100, 1000)},{np.random.randint(0, 100):02d}"
    return f"{np.random.randint(10, 100):02d},{np.random.randint(0, 100):02d}"


def load_font(font_path, font_size):
    """Load a font, fallback to default if loading fails."""
    try:
        return ImageFont.truetype(font_path, font_size)
    except IOError:
        logging.error(f"Cannot load font at {font_path}. Using default font.")
        return ImageFont.load_default()


# Tag dictionary
def prepare_tag_dict(all_possible_items):
    return {
        "#13number#": lambda: get_random_numbers(13),
        "#12number#": lambda: get_random_numbers(12),
        "#10number#": lambda: get_random_numbers(10),
        "#date#": get_random_date,  # dd.mm.yy
        "#time#": get_random_time,  # hh:mm:ss
        "#item#": lambda: get_random_item(all_possible_items),
        "#comma_number#": get_random_comma_number,
    }


# Overlap checking functions
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Compute area of intersection
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Compute area of both boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute IoU
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    iou = inter_area / union_area
    return iou


def is_significant_overlap(box1, box2, iou_threshold=0.1):
    """Check if two boxes overlap more than the IoU threshold."""
    iou = calculate_iou(box1, box2)
    return iou > iou_threshold


def validate_bb(bb, min_width=1, min_height=1):
    """
    Ensure that the bounding box (x, y, w, h) has at least min_width and min_height.
    If not, adjust w and/or h accordingly.
    """
    x, y, w, h = bb
    if w < min_width:
        w = min_width
    if h < min_height:
        h = min_height
    return (x, y, w, h)


def apply_random_offsets(bb, max_horizontal_offset, max_vertical_offset, existing_bbs):
    """Apply random offsets to the bounding box without significant overlaps."""
    x, y, w, h = bb
    for attempt in range(100):
        offset_x = np.random.randint(-max_horizontal_offset, max_horizontal_offset + 1)
        offset_y = np.random.randint(-max_vertical_offset, max_vertical_offset + 1)
        new_bb = (x + offset_x, y + offset_y, w, h)

        # Optionally, add image boundary checks here


        # Check for significant overlaps using IoU
        if not any(is_significant_overlap(new_bb, other_bb) for other_bb in existing_bbs):
            return new_bb
    # If no suitable position found, return the original bounding box to minimize overlaps
    logging.warning(
        f"Could not find non-overlapping position after 100 attempts for box {bb}. Using original position.")
    return bb


# Text fitting functions
def fit_text_in_box(img, text, box, font_path, initial_font_size, fill=(0, 0, 0), show_BB=False):
    draw = ImageDraw.Draw(img)
    x, y, w, h = box

    # Try decreasing font size until the text fits
    font_size = initial_font_size
    while font_size > 5:  # set a reasonable lower bound for font size
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            # Fall back to default font if truetype font can't be loaded
            font = ImageFont.load_default()
        text_width = draw.textlength(text, font=font)
        bbox = font.getbbox(text)
        text_height = bbox[3] - bbox[1] if bbox else font.getsize(text)[1]

        if text_width <= w and text_height <= h:
            break
        font_size -= 1

    if font_size <= 5:
        # If text still does not fit, log a warning and skip drawing.
        logging.warning(f"Text '{text}' cannot fit in the box {box} even with reduced font size.")
        return

    # Compute top-left corner so that text is centered in box
    start_x = x + (w - text_width) // 2
    start_y = y + (h - text_height) // 2

    draw.text((start_x, start_y), text, font=font, fill=fill)

    if show_BB:
        draw.rectangle([x, y, x + w, y + h], outline="red", width=1)


# Image augmentation function (placeholder)
def apply_augmentations(image):
    # Implement your augmentation logic here
    # For example, flipping, rotating, adding noise, etc.
    return image


# Validate and load all possible items
all_possible_items = validate_all_possible_items("all_possible_items.txt")

# Prepare tag dictionary
tag_dict = prepare_tag_dict(all_possible_items)


# Main synthetic dataset generation
def generate_synthetic_dataset(
        image_path,
        output_dir,
        bbs,
        items_with_tags,
        font_dir,
        num_receipts=5,
        font_size=14,
        show_BB=False
):
    """
    Generate receipts by placing each item exactly in the bounding boxes
    with no random offset, ensuring no overlap.
    """
    # Load the base image (receipt background)
    base_image = Image.open(image_path).convert("RGB")

    # Load available fonts
    font_files = [f for f in os.listdir(font_dir) if f.lower().endswith(('.ttf', '.otf'))]
    if not font_files:
        raise ValueError(f"No valid fonts found in {font_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    data = []

    for i in tqdm(range(num_receipts), desc="Generating Receipts"):
        # Copy the base image
        receipt_img = base_image.copy()

        # Randomly pick a font, or just pick one for consistency
        chosen_font = os.path.join(font_dir, np.random.choice(font_files))

        strings_used = []
        all_bbs = []

        # Make sure we have as many bounding boxes as items
        # We'll simply place each item in each corresponding bounding box
        for idx, (bb) in enumerate(bbs[:len(items_with_tags)]):
            text_to_place = items_with_tags[idx]

            # If using tags, do your replacement here
            # Example: If text starts with "#" and ends with "#", do a dictionary lookup
            # ...

            strings_used.append(text_to_place)
            fit_text_in_box(
                receipt_img,
                text=text_to_place,
                box=bb,
                font_path=chosen_font,
                font_size=font_size,
                show_BB=show_BB
            )
            all_bbs.append(bb)

        # Optionally apply your image augmentations here:
        # receipt_img = apply_augmentations(receipt_img)

        # Save final image
        out_path = os.path.join(output_dir, f"receipt_{i + 1}.jpg")
        receipt_img.save(out_path)

        # Prepare JSON data
        record_bbs = []
        for text_val, (bx, by, bw, bh) in zip(strings_used, all_bbs):
            record_bbs.append([text_val, bx, by, bw, bh])

        data.append({
            "image_path": out_path,
            "strings_and_bbs": record_bbs
        })

    # Finally, save a JSON with all bounding boxes
    json_path = os.path.join(output_dir, "labels.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

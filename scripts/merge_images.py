import argparse
import os
from PIL import Image

def merge_images_horizontally(image_paths, output_path):
    valid_paths = [p for p in image_paths if p is not None]

    for path in valid_paths:
        if not os.path.exists(path):
            print(f"Błąd: Plik '{path}' nie istnieje.")
            return

    images = [Image.open(path) for path in valid_paths]
    base_height = images[0].height
    resized_images = []

    for img in images:
        if img.height != base_height:
            new_width = int(img.width * (base_height / img.height))
            img_resized = img.resize((new_width, base_height), Image.Resampling.LANCZOS)
            resized_images.append(img_resized)
        else:
            resized_images.append(img)

    total_width = sum(img.width for img in resized_images)

    combined_image = Image.new(
        "RGBA" if any(img.mode == "RGBA" for img in resized_images) else "RGB",
        (total_width, base_height)
    )

    x_offset = 0
    for img in resized_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined_image.save(output_path)
    print(f"Sukces! Połączony obraz zapisano jako: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Łączy obrazy w poziomie od lewej do prawej."
    )
    parser.add_argument("img1", help="Ścieżka do pierwszego obrazu")
    parser.add_argument("img2", help="Ścieżka do drugiego obrazu")
    parser.add_argument("img3", help="Ścieżka do trzeciego obrazu")
    parser.add_argument("img4", nargs="?", default=None, help="Ścieżka do czwartego obrazu (opcjonalny)")
    parser.add_argument("-o", "--output", default="merged_output.png", help="Nazwa wyjściowego pliku")

    args = parser.parse_args()

    image_paths = [args.img1, args.img2, args.img3, args.img4]
    merge_images_horizontally(image_paths, args.output)

if __name__ == "__main__":
    main()
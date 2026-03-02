from src.monitor.log_usage import log_usage
import argparse
import json
import matplotlib.pyplot as plt

from src.inference_surface_cgan import generate


CLASS_MAP_PATH = "data/processed/class_map.json"


def load_class_map():

    with open(CLASS_MAP_PATH, "r") as f:
        return json.load(f)


def get_class_index(name, class_map):

    for k, v in class_map.items():

        if v == name:
            return int(k)

    return None


def main():

    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--class",
        dest="cls_name",
        required=True,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    class_map = load_class_map()

    cls = get_class_index(args.cls_name, class_map)

    if cls is None:

        print("Class not found")
        print(class_map)

        return

    imgs = generate(cls, args.n)

    log_usage(
        source="cli",
        cls=args.cls_name,
        n=args.n,
    )

    for i, img in enumerate(imgs):

        plt.subplot(1, args.n, i + 1)

        plt.imshow(img)

        plt.axis("off")

    plt.show()


if __name__ == "__main__":

    main()
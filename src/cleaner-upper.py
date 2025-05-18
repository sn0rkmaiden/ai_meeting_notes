from sys import argv
import glob
import json


def cleanup():
    root = argv[1]
    json_files = glob.glob(root + "/**/*.json", recursive=True)
    for file in json_files:
        with open(file) as fd:
            read_json = json.load(fd)
            for proj in read_json:
                for ann in proj["annotations"]:
                    ann["result"] = [line for line in ann["result"] if line["value"]["start"] < line["original_length"]]
                if "prediction" in proj:
                    proj["prediction"] = [line for line in proj["prediction"] if line["value"]["start"] < line["original_length"]]
        with open(file, "w") as fd:
            json.dump(read_json, fd, ensure_ascii=False)


if __name__ == "__main__":
    cleanup()

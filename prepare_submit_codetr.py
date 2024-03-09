import datetime as dt
import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm


def get_file_name_by_time(ext="json") -> str:
    name = dt.datetime.utcnow().isoformat()
    name = "".join([(e.isalnum() and e or "_") for e in name])
    return name + "." + ext


def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId


def build_args():
    parser = ArgumentParser()
    parser.add_argument("json_path", type=str, help="json file")
    return parser.parse_args()


def main():
    args = build_args()

    input_json = Path(args.json_path)
    output_submit = Path(f'submissions/{get_file_name_by_time()}')
    output_submit.parent.mkdir(parents=True, exist_ok=True)

    json_list = input_json.glob('*.json')

    submit = []
    conf_thes = 0
    for j in tqdm(json_list):
        name = j.stem
        file = json.load(open(j, 'r'))
        scores = file['scores']
        labels = file['labels']
        boxes = file['bboxes']
        for i, score in enumerate(scores):
            if score > conf_thes:
                submit.append({
                    "image_id": get_image_Id(name),
                    "category_id": int(labels[i]),
                    "bbox": [boxes[i][0], boxes[i][1], boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1]],
                    "score": round(float(score), 5)
                })

    with open(output_submit, 'w') as f:
        json.dump(submit, f)

    print("Result saved to ", output_submit)


if __name__ == "__main__":
    main()

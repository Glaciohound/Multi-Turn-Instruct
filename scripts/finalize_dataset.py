import argparse
import json
import os
import hashlib

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, default="data/sources/")
parser.add_argument('--output_dir', type=str, default="data/dataset/")
parser.add_argument('--data-composition', type=str, nargs='+', default=["all"])
args = parser.parse_args()

data_composition = dict()
if args.data_composition[0] == "all":
    assert len(args.data_composition) == 1
else:
    for component in args.data_composition:
        data_composition[component.split(":")[0]] = \
            int(component.split(":")[1])

all_hashes = set()

output_dataset = dict()
metadata = dict()
tasks = os.listdir(args.source_dir)
for task in tasks:
    task_dir = os.path.join(args.source_dir, task)
    if os.path.isfile(task_dir):
        continue
    subtasks = os.listdir(task_dir)
    output_dataset[task] = dict()
    metadata[task] = dict()
    for subtask in subtasks:
        subtask_dir = os.path.join(task_dir, subtask)
        if os.path.isfile(subtask_dir):
            continue
        with open(os.path.join(subtask_dir, "metric.json"), 'r') as f:
            metric = json.load(f)
        with open(os.path.join(subtask_dir, "constructed.json"), 'r') as f:
            source_data = json.load(f)
        subtask_size = data_composition.get(
            f"{task}-{subtask}", len(source_data))
        for datum_id, datum in list(source_data.items())[:subtask_size]:
            # id_hash = hex(hash(f"{task}-{subtask}-{datum_id}"))[-8:]
            id_hash = hashlib.md5(
                f"{task}-{subtask}-{datum_id}".encode()
            ).hexdigest()[:8]
            assert id_hash not in all_hashes
            all_hashes.add(id_hash)
            for turn in datum:
                if turn["role"] == "assistant":
                    turn.update(metric)
            metadatum = {
                "task": task,
                "subtask": subtask,
                "datum_id": datum_id,
            }
            metadatum.update(metric)
            output_dataset[task][id_hash] = datum
            metadata[task][id_hash] = metadatum
        print(f"Task {task} Subtask {subtask} Size {subtask_size}")
    print(f"Task {task} Total Size {len(output_dataset[task])}\n\n")

dataset_file = os.path.join(args.output_dir, "dataset.json")
metadata_file = os.path.join(args.output_dir, "metadata.json")
with open(dataset_file, 'w') as f:
    json.dump(output_dataset, f, indent=4)
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=4)

import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--prediction_dir', type=str, required=True)
parser.add_argument('--metadata-file', type=str, required=True)
args = parser.parse_args()

predictions_file = f"{args.prediction_dir}/predictions.json"

with open(predictions_file, 'r') as f:
    predictions = json.load(f)
with open(args.metadata_file, 'r') as f:
    metadata = json.load(f)


for task, task_scores in predictions.items():
    # sanity check
    for datum_score in task_scores.values():
        for turn_score in datum_score:
            assert (turn_score["score"] is None and
                    turn_score["evaluation_reference"] is None) or \
                (turn_score["score"] is not None and
                 turn_score["evaluation_reference"] is not None)

    # task-wise scores
    task_score = np.mean([
        turn_score["score"]
        for datum_score in task_scores.values()
        for turn_score in datum_score
        if turn_score["score"] is not None
    ])
    print(f"Task: {task}, average score: {task_score}")

    # turn-wise scores
    turn_wise_scores = [
        [0, 0] for _ in range(40)
    ]
    for datum_score in task_scores.values():
        for turn_score in datum_score:
            if turn_score["score"] is None:
                continue
            turn_id = turn_score["turn_id"] // 2
            turn_wise_scores[turn_id][0] += turn_score["score"]
            turn_wise_scores[turn_id][1] += 1
    turn_wise_scores = [
        (score / count, count) if count > 0 else None
        for score, count in turn_wise_scores
    ]
    print("Turn-wise scores:")
    print(list(enumerate(turn_wise_scores)))

    # subtask-wise scores
    score_breakdown = dict()
    for datum_id, score in task_scores.items():
        datum_info = metadata[task][datum_id]
        subtask = datum_info['subtask']
        if subtask not in score_breakdown:
            score_breakdown[subtask] = []
        score_breakdown[subtask].extend(
            [turn_score["score"] for turn_score in score
             if turn_score["score"] is not None]
        )
    score_breakdown = {
        subtask: np.mean(scores)
        for subtask, scores in score_breakdown.items()
    }
    for subtask, score in score_breakdown.items():
        print(f"{subtask}: {score}", end=", ")
    print()

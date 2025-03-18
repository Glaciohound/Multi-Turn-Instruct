import argparse
import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

from models.get_model import get_model
from evaluation.metric_functions import metric_function_dict


def parse_args():
    def none_or_int(value):
        if value == 'None':
            return None
        return int(value)

    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--openai-key', type=str, default=None)
    parser.add_argument('--region', type=str, default="us-east-1")
    parser.add_argument('--use-ground-truth-context', action='store_true')
    # data
    parser.add_argument('--dataset-file', type=str, required=True)
    # evaluation
    parser.add_argument('--turn-window', type=int, default=None)
    parser.add_argument('--max-context-size', type=none_or_int, default=None)
    parser.add_argument('--max-tokens-per-turn', type=int, default=1024)
    # output
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    return args


def get_model_and_dataset(args):
    model = get_model(
        args.model, args.openai_key, args.region
    )
    with open(args.dataset_file, 'r') as f:
        dataset = json.load(f)
    return model, dataset


def setup_files(args):
    if args.turn_window is None:
        output_dir = f"{args.output_dir}/{args.model}"
    else:
        output_dir = f"{args.output_dir}/"\
            f"{args.model}_turn_window_{args.turn_window}"
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = f"{output_dir}/predictions.json"
    scores_file = f"{output_dir}/scores.json"
    summary_file = f"{output_dir}/summary.json"
    return predictions_file, scores_file, summary_file


def print_eval_info(model, predictions_file, scores_file, summary_file):
    print(f"Model: {model}")
    print(f"Predictions file: {predictions_file}")
    print(f"Scores file: {scores_file}")
    print(f"Summary file: {summary_file}")


def initialize_predictions(predictions_file):
    predictions = defaultdict(dict)
    if os.path.exists(predictions_file):
        print("Loading existing predictions")
        with open(predictions_file, 'r') as f:
            predictions.update(json.load(f))
    return predictions


def log_and_summarize(
        predictions,
        predictions_file, scores_file, summary_file):
    scores = {
        task: {
            datum_id: [turn["score"] for turn in datum
                       if turn["score"] is not None]
            for datum_id, datum in task_predictions.items()
        }
        for task, task_predictions in predictions.items()
    }
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=4)
    with open(scores_file, 'w') as f:
        json.dump(scores, f, indent=4)
    summary = {
        task: np.mean([
                score
                for datum_score in task_scores.values()
                for score in datum_score
            ])
        for task, task_scores in scores.items()
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    return summary


def get_predictions(datum, max_context_size):
    datum_predictions = []
    for turn_id, turn in enumerate(datum):
        if turn["role"] == "user":
            continue
        else:
            if args.use_ground_truth_context and \
                    turn["evaluation_reference"] is None:
                continue
        context = model.compose_context(
            deepcopy(datum[:turn_id]),
            datum_predictions,
            use_ground_truth_context=args.use_ground_truth_context
        )
        if args.turn_window is not None:
            context = context[-args.turn_window:]
        try:
            response = model.respond(
                context,
                max_tokens=args.max_tokens_per_turn,
                max_context_size=max_context_size
            )
        except Exception as e:
            if "length" in str(e):
                response = "ERROR"
            else:
                raise e
        datum_predictions.append({
            "turn_id": turn_id,
            "ground_truth": turn["text"],
            "instruction": datum[turn_id-1]["text"],
            "response": response,
            "evaluation_reference": turn["evaluation_reference"],
            "evaluation_metric": turn["evaluation_metric"],
        })
    return datum_predictions


def score_datum_predictions(datum_predictions):
    for turn in datum_predictions:
        response = turn["response"]
        evaluation_metric = turn["evaluation_metric"]
        evaluation_reference = turn["evaluation_reference"]
        score = metric_function_dict[evaluation_metric](
            response, evaluation_reference)
        turn["score"] = score
        print(score, evaluation_reference, response)


def evaluate(model, dataset, predictions_file):
    predictions = initialize_predictions(predictions_file)
    for task, task_dataset in dataset.items():
        print("Evaluating:", task)
        pbar = tqdm(list(task_dataset.items()), total=len(task_dataset))
        for datum_id, datum in pbar:
            if datum_id not in predictions[task].keys():
                datum_predictions = get_predictions(
                    datum, args.max_context_size)
                score_datum_predictions(datum_predictions)
                predictions[task][datum_id] = datum_predictions
                print(log_and_summarize(
                    predictions, predictions_file, scores_file, summary_file))
            else:
                print("Skipping", task, datum_id)
                score_datum_predictions(predictions[task][datum_id])
                print(log_and_summarize(
                    predictions, predictions_file, scores_file, summary_file))


if __name__ == "__main__":
    args = parse_args()
    model, dataset = get_model_and_dataset(args)
    predictions_file, scores_file, summary_file = setup_files(args)
    print_eval_info(args.model, predictions_file, scores_file, summary_file)
    evaluate(model, dataset, predictions_file)
    print_eval_info(args.model, predictions_file, scores_file, summary_file)

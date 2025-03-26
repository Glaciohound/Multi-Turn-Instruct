import re
import string
from .squad_metric import squad_metric
from nltk.translate.bleu_score import sentence_bleu


def matching_any(response, references):
    if references is None:
        return None
    return float(any(piece.lower() in response.lower()
                     for piece in references))


def not_matching_any(response, references):
    if references is None:
        return None
    return 1 - matching_any(response, references)


def matching_any_exact(response, references):
    if references is None:
        return None
    # v0
    # answer = re.search(r"Answer: (.+)", response)
    # if answer is None:
    #     return 0
    # answer = answer.groups()[0].strip()
    # v1
    # answer = re.findall(r"Answer: (.+?)", response)
    # if len(answer) == 0:
    #     return 0
    # answer = answer[-1].strip()
    # v2
    match = re.search(r"Answer:\s*(.+)", response)
    if not match:
        return 0
    answer = match.group(1).strip()
    answer = answer.rstrip(string.punctuation)
    score = float(any(answer.lower() == piece.lower()
                      for piece in references))
    if score < 1:
        print(answer, references, score)
    return score


def intersection_over_union(response, references):
    if references is None:
        return None
    response = response.lower().strip()
    if ": " in response:
        response = response.split(": ")[1]
    response = set(response.split(", "))
    references = set([piece.lower() for piece in references])
    matching_count = 0
    for piece in references:
        if piece in response:
            matching_count += 1
    return matching_count / (len(response) + len(references) - matching_count)


def bleu_metric(response, references):
    if references is None:
        return None
    return sentence_bleu([piece.split() for piece in references],
                         response.split())


metric_function_dict = {
    "matching_any": matching_any,
    "not_matching_any": not_matching_any,
    "matching_any_exact": matching_any_exact,
    "intersection_over_union": intersection_over_union,
    "squad": squad_metric,
    "bleu": bleu_metric,
}

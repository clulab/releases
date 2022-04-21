import torch
from torch import nn


def margin_loss_two_way(scores_per_step, margin):
    """
    Args:
        scores_per_step:
            The scores: A list of tensors. The tensors are not necessarily of the same size.
        margin:
            The margin to be used in the nn.MarginRankingLoss
    """
    """assumes first score is correct score"""
    correct = []
    incorrect = []
    for scores in scores_per_step:
        # next correct should have higher score than prev correct and next incorrects
        correct_score = scores[0]
        incorrect_scores = scores[1:]
        correct_scores = correct_score.expand(len(incorrect_scores), 1)
        correct.append(correct_scores)
        incorrect.append(incorrect_scores)
        # prev correct should have higher score than next incorrects
        correct_score = scores[1]
        incorrect_scores = scores[2:]
        correct_scores = correct_score.expand(len(incorrect_scores), 1)
        correct.append(correct_scores)
        incorrect.append(incorrect_scores)
    # calculate loss
    loss_fn = nn.MarginRankingLoss(margin)
    correct = torch.cat(correct)
    incorrect = torch.cat(incorrect)
    target = torch.ones_like(correct)
    loss = loss_fn(correct, incorrect, target)
    return loss

def margin_loss_one_way(scores_per_step, margin):
    """
    Args:
        scores_per_step:
            The scores: A list of tensors. The tensors are not necessarily of the same size.
        margin:
            The margin to be used in the nn.MarginRankingLoss
    """
    """assumes first score is correct score"""
    correct = []
    incorrect = []
    for scores in scores_per_step:
        correct_score = scores[0]
        incorrect_scores = scores[1:]
        # repeat correct score for the pairwise comparison
        correct_scores = correct_score.expand(len(incorrect_scores), 1)
        correct.append(correct_scores)
        incorrect.append(incorrect_scores)
    # calculate loss
    loss_fn = nn.MarginRankingLoss(margin)
    correct = torch.cat(correct)
    incorrect = torch.cat(incorrect)
    target = torch.ones_like(correct)
    loss = loss_fn(correct, incorrect, target)
    return loss
    
"""Auto-generated from the CTGP research monolith — do not edit by hand."""

import copy
import math
import random
import numpy as np
from ctgp_lib.config import *

# ============================================================================
# CTGP genotype operations (from graph.py)
# ============================================================================

def _random_gp_leaf():
    return {
        'type': 'leaf',
        'conv': random.choice(CONV_NAMES),
        'heads': random.choice(HEAD_OPTIONS),
        'hidden': random.randint(HID_MIN, HID_MAX)
    }


def _random_gp_program():
    n_leaves = random.randint(MIN_LAYERS, MAX_LAYERS)
    nodes = [_random_gp_leaf() for _ in range(n_leaves)]
    while len(nodes) > 1:
        a = nodes.pop(random.randrange(len(nodes)))
        b = nodes.pop(random.randrange(len(nodes)))
        nodes.append({'type': 'concat', 'left': a, 'right': b})
    return {
        'type': 'program',
        'tree': nodes[0],
        'drop': random.uniform(DROP_MIN, DROP_MAX),
        'lr': 10 ** random.uniform(math.log10(LR_MIN), math.log10(LR_MAX)),
        'init': random.choice(INIT_OPTIONS),
        'cls_hidden': [random.randint(HID_MIN, HID_MAX) for _ in range(random.randint(1, MAX_CLS_LAYERS))]
    }


def _collect_leaves(node, out):
    if node['type'] == 'leaf':
        out.append(node)
        return
    _collect_leaves(node['left'], out)
    _collect_leaves(node['right'], out)


def _gp_to_solution(program):
    leaves = []
    _collect_leaves(program['tree'], leaves)
    leaves = leaves[:MAX_LAYERS]
    if not leaves:
        leaves = [_random_gp_leaf()]
    return {
        'conv': [l['conv'] for l in leaves],
        'heads': [l['heads'] for l in leaves],
        'hidden': [l['hidden'] for l in leaves],
        'drop': program['drop'],
        'lr': program['lr'],
        'init': program['init'],
        'cls_hidden': program['cls_hidden'].copy()
    }


def _collect_paths(node, current_path, out):
    out.append(current_path)
    if node['type'] == 'concat':
        _collect_paths(node['left'], current_path + ['left'], out)
        _collect_paths(node['right'], current_path + ['right'], out)


def _get_subtree(tree, path):
    node = tree
    for step in path:
        node = node[step]
    return node


def _set_subtree(tree, path, subtree):
    if not path:
        return subtree
    root = copy.deepcopy(tree)
    node = root
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = subtree
    return root


def _gp_mutate(program):
    p = copy.deepcopy(program)
    # Favor layer variation (conv type + hidden/heads per layer)
    move = random.choice(['mutate_leaf'] * 4 + ['replace_subtree', 'wrap_concat', 'mutate_hyper'] * 2)

    if move == 'mutate_leaf':
        leaves = []
        _collect_leaves(p['tree'], leaves)
        if not leaves:
            return p
        leaf = random.choice(leaves)
        # Weights: 40% conv type change (heterogeneity), 30% hidden, 30% heads
        field = random.choices(['conv', 'hidden', 'heads'], weights=[0.4, 0.3, 0.3], k=1)[0]
        if field == 'conv':
            # Change to DIFFERENT conv type (not same)
            candidates = [c for c in CONV_NAMES if c != leaf['conv']]
            leaf['conv'] = random.choice(candidates) if candidates else leaf['conv']
        elif field == 'heads':
            leaf['heads'] = random.choice(HEAD_OPTIONS)
        else:
            # Hidden: small random walk for diversity
            delta = random.randint(-32, 32)
            leaf['hidden'] = int(np.clip(leaf['hidden'] + delta, HID_MIN, HID_MAX))
        return p

    if move == 'mutate_hyper':
        target = random.choice(['drop', 'lr', 'init', 'cls_hidden'])
        if target == 'drop':
            d = (DROP_MAX - DROP_MIN) * 0.1
            p['drop'] = float(np.clip(p['drop'] + random.uniform(-d, d), DROP_MIN, DROP_MAX))
        elif target == 'lr':
            log_lr = math.log10(p['lr'])
            d = (math.log10(LR_MAX) - math.log10(LR_MIN)) * 0.1
            p['lr'] = 10 ** np.clip(log_lr + random.uniform(-d, d), math.log10(LR_MIN), math.log10(LR_MAX))
        elif target == 'init':
            p['init'] = random.choice(INIT_OPTIONS)
        else:
            if len(p['cls_hidden']) > 1 and random.random() < 0.5:
                p['cls_hidden'].pop()
            elif len(p['cls_hidden']) < MAX_CLS_LAYERS:
                p['cls_hidden'].append(random.randint(HID_MIN, HID_MAX))
            else:
                i = random.randrange(len(p['cls_hidden']))
                p['cls_hidden'][i] = random.randint(HID_MIN, HID_MAX)
        return p

    paths = []
    _collect_paths(p['tree'], [], paths)
    target = random.choice(paths)
    if move == 'replace_subtree':
        p['tree'] = _set_subtree(p['tree'], target, _random_gp_program()['tree'])
        return p

    subtree = _get_subtree(p['tree'], target)
    wrapped = {
        'type': 'concat',
        'left': copy.deepcopy(subtree),
        'right': _random_gp_leaf()
    }
    p['tree'] = _set_subtree(p['tree'], target, wrapped)
    return p

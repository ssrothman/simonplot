import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import (Patch, Rectangle, Shadow, FancyBboxPatch,
                                StepPatch)
from matplotlib.text import Text
from matplotlib.collections import (
    Collection, CircleCollection, LineCollection, PathCollection,
    PolyCollection, RegularPolyCollection)

options = [
    (0.05, 0.95, 'top', 'left'),
    (0.95, 0.95, 'top', 'right'),

    (0.05, 0.05, 'bottom', 'left'),
    (0.95, 0.05, 'bottom', 'right'),

    (0.5, 0.95, 'top', 'center'),
    (0.5, 0.05, 'bottom', 'center'),

    (0.05, 0.5, 'center', 'left'),
    (0.95, 0.5, 'center', 'right'),

    (0.5, 0.5, 'center', 'center'),
]

option_names = [
    'top-left',
    'top-right',
    'bottom-left',
    'bottom-right',
    'top-center',
    'bottom-center',
    'center-left',
    'center-right',
    'center-center',
]

def get_other_objects(ax):
    lines = []
    offsets = []
    bboxes = []

    for artist in ax.get_children():
        if isinstance(artist, Line2D):
            lines.append(
                artist.get_transform().transform_path(artist.get_path()))
        elif isinstance(artist, Rectangle):
            bboxes.append(
                artist.get_bbox().transformed(artist.get_data_transform()))
        elif isinstance(artist, Patch):
            lines.append(
                artist.get_transform().transform_path(artist.get_path()))
        elif isinstance(artist, PolyCollection):
            lines.extend(artist.get_transform().transform_path(path)
                            for path in artist.get_paths())
        elif isinstance(artist, Collection):
            transform, transOffset, hoffsets, _ = artist._prepare_points() # pyright: ignore[reportAttributeAccessIssue]
            if len(hoffsets):
                offsets.extend(transOffset.transform(hoffsets))
        elif isinstance(artist, Text):
            bboxes.append(artist.get_window_extent(renderer=ax.figure.canvas.get_renderer()))

    return lines, offsets, bboxes

def get_text_bbox(ax, text, option, fontsize=24, bbox_opts={}):
    fig = ax.figure
    tmp_txt = ax.text(
        option[0], option[1], text,
        horizontalalignment=option[3],
        verticalalignment=option[2],
        transform=ax.transAxes,
        fontsize=fontsize,
        bbox=bbox_opts
    )
    fig.canvas.draw()
    bbox = tmp_txt.get_window_extent(renderer=fig.canvas.get_renderer())
    tmp_txt.remove()
    return bbox

def compute_overlap_area(bbox1, bbox2):
    x_left = max(bbox1.x0, bbox2.x0)
    y_top = max(bbox1.y0, bbox2.y0)
    x_right = min(bbox1.x1, bbox2.x1)
    y_bottom = min(bbox1.y1, bbox2.y1)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)

def place_text(ax, text, loc, fontsize=24, bbox_opts={}):
    #attempt to copy the "best" location logic from matplotlib's legend
    badnesses = []

    other_lines, other_offsets, other_bboxes = get_other_objects(ax)

    if loc == 'best':
        for opt in options:
            #print("Attempting option", len(badnesses))
            text_bbox = get_text_bbox(ax, text, opt, fontsize, bbox_opts)

            badness = 0

            for ol in other_lines:
                badness += text_bbox.count_contains(ol.vertices) 
                badness += ol.intersects_bbox(text_bbox)

            badness += text_bbox.count_contains(other_offsets)

            badness += text_bbox.count_overlaps(other_bboxes)

            badnesses.append(badness)
            #print("\tBadness:", badness)

        best_index = np.argmin(badnesses)
        best_option = options[best_index]

    else:
        if type(loc) is str:
            if loc not in option_names:
                raise ValueError("loc string not recognized")
            best_option = options[option_names.index(loc)]
        else:
            best_option = loc

    return ax.text(
        best_option[0], best_option[1], text,
        horizontalalignment=best_option[3],
        verticalalignment=best_option[2],
        transform=ax.transAxes,
        fontsize=fontsize,
        bbox=bbox_opts
    )

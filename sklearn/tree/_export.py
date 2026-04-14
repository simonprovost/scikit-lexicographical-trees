"""
This module defines export functions for decision trees.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Iterable
from io import StringIO
from numbers import Integral, Real

import numpy as np

from sklearn.base import is_classifier
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
    validate_params,
)
from sklearn.utils.validation import check_array, check_is_fitted

from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from ._reingold_tilford import Tree, buchheim


def _color_brew(n):
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360.0 / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


class Sentinel:
    def __repr__(self):
        return '"tree.dot"'


SENTINEL = Sentinel()


@validate_params(
    {
        "decision_tree": [DecisionTreeClassifier, DecisionTreeRegressor],
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],
        "feature_names": ["array-like", None],
        "class_names": ["array-like", "boolean", None],
        "label": [StrOptions({"all", "root", "none"})],
        "filled": ["boolean"],
        "impurity": ["boolean"],
        "node_ids": ["boolean"],
        "proportion": ["boolean"],
        "rounded": ["boolean"],
        "precision": [Interval(Integral, 0, None, closed="left"), None],
        "ax": "no_validation",  # delegate validation to matplotlib
        "fontsize": [Interval(Integral, 0, None, closed="left"), None],
        "tpt_time_scale": [Interval(Real, 0, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)
def plot_tree(
    decision_tree,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    filled=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rounded=False,
    precision=3,
    ax=None,
    fontsize=None,
    tpt_time_scale=None,
):
    """Plot a decision tree.

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    The visualization is fit automatically to the size of the axis.
    Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
    the size of the rendering.

    Read more in the :ref:`User Guide <tree>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    decision_tree : decision tree regressor or classifier
        The decision tree to be plotted.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : array-like of str, default=None
        Names of each of the features.
        If None, generic names will be used ("x[0]", "x[1]", ...).

    class_names : array-like of str or True, default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, default=False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    impurity : bool, default=True
        When set to ``True``, show the impurity at each node.

    node_ids : bool, default=False
        When set to ``True``, show the ID number on each node.

    proportion : bool, default=False
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rounded : bool, default=False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, default=3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    ax : matplotlib axis, default=None
        Axes to plot to. If None, use current axis. Any previous content
        is cleared.

    fontsize : int, default=None
        Size of text font. If None, determined automatically to fit figure.

    tpt_time_scale : float, default=None
        Scale factor for time distance to vertical distance in Time-penalized Trees (TpT).
        Used to adjust vertical spacing based on time differences between nodes.
        If None, defaults to 0.5. Only relevant for TpT trees.

    Returns
    -------
    annotations : list of artists
        List containing the artists for the annotation boxes making up the
        tree.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.plot_tree(clf)
    [...]
    """

    check_is_fitted(decision_tree)

    exporter = _MPLTreeExporter(
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        label=label,
        filled=filled,
        impurity=impurity,
        node_ids=node_ids,
        proportion=proportion,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
        tpt_time_scale=tpt_time_scale,
    )
    return exporter.export(decision_tree, ax=ax)


class _BaseTreeExporter:
    def __init__(
        self,
        max_depth=None,
        feature_names=None,
        class_names=None,
        label="all",
        filled=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        fontsize=None,
    ):
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.class_names = class_names
        self.label = label
        self.filled = filled
        self.impurity = impurity
        self.node_ids = node_ids
        self.proportion = proportion
        self.rounded = rounded
        self.precision = precision
        self.fontsize = fontsize

    def get_color(self, value):
        # Find the appropriate color & intensity for a node
        if self.colors["bounds"] is None:
            # Classification tree
            color = list(self.colors["rgb"][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0.0
            else:
                alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])
        else:
            # Regression tree or multi-output
            color = list(self.colors["rgb"][0])
            alpha = (value - self.colors["bounds"][0]) / (
                self.colors["bounds"][1] - self.colors["bounds"][0]
            )
        # compute the color as alpha against white
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return "#%2x%2x%2x" % tuple(color)

    def get_fill_color(self, tree, node_id):
        # Fetch appropriate color for node
        if "rgb" not in self.colors:
            # Initialize colors and bounds if required
            self.colors["rgb"] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                # Find max and min impurities for multi-output
                self.colors["bounds"] = (np.min(-tree.impurity), np.max(-tree.impurity))
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                # Find max and min values in leaf nodes for regression
                self.colors["bounds"] = (np.min(tree.value), np.max(tree.value))
        if tree.n_outputs == 1:
            node_val = tree.value[node_id][0, :]
            if (
                tree.n_classes[0] == 1
                and isinstance(node_val, Iterable)
                and self.colors["bounds"] is not None
            ):
                # Unpack the float only for the regression tree case.
                # Classification tree requires an Iterable in `get_color`.
                node_val = node_val.item()
        else:
            # If multi-output color node by impurity
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)

    def node_to_str(self, tree, node_id, criterion):
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (self.label == "root" and node_id == 0) or self.label == "all"

        characters = self.characters
        node_string = characters[-1]

        # Write node ID
        if self.node_ids:
            if labels:
                node_string += "node "
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = "x%s%s%s" % (
                    characters[1],
                    tree.feature[node_id],
                    characters[2],
                )
            node_string += "%s %s %s%s" % (
                feature,
                characters[3],
                round(tree.threshold[node_id], self.precision),
                characters[4],
            )

        # Write impurity
        if self.impurity:
            if isinstance(criterion, _criterion.FriedmanMSE):
                criterion = "friedman_mse"
            elif isinstance(criterion, _criterion.MSE) or criterion == "squared_error":
                criterion = "squared_error"
            elif not isinstance(criterion, str):
                criterion = "impurity"
            if labels:
                node_string += "%s = " % criterion
            node_string += (
                str(round(tree.impurity[node_id], self.precision)) + characters[4]
            )

        # Write node sample count
        if labels:
            node_string += "samples = "
        if self.proportion:
            percent = (
                100.0 * tree.n_node_samples[node_id] / float(tree.n_node_samples[0])
            )
            node_string += str(round(percent, 1)) + "%" + characters[4]
        else:
            node_string += str(tree.n_node_samples[node_id]) + characters[4]

        # Write node class distribution / regression value
        if not self.proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value * tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += "value = "
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, self.precision)
        elif self.proportion:
            # Classification
            value_text = np.around(value, self.precision)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
        else:
            # Classification with floating-point weights
            value_text = np.around(value, self.precision)
        # Strip whitespace
        value_text = str(value_text.astype("S32")).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]

        # Write node majority class
        if (
            self.class_names is not None
            and tree.n_classes[0] != 1
            and tree.n_outputs == 1
        ):
            # Only done for single-output classification trees
            if labels:
                node_string += "class = "
            if self.class_names is not True:
                class_name = self.class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (
                    characters[1],
                    np.argmax(value),
                    characters[2],
                )
            node_string += class_name

        # Clean up any trailing newlines
        if node_string.endswith(characters[4]):
            node_string = node_string[: -len(characters[4])]

        return node_string + characters[5]


class _DOTTreeExporter(_BaseTreeExporter):
    def __init__(
        self,
        out_file=SENTINEL,
        max_depth=None,
        feature_names=None,
        class_names=None,
        label="all",
        filled=False,
        leaves_parallel=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rotate=False,
        rounded=False,
        special_characters=False,
        precision=3,
        fontname="helvetica",
    ):
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
        )
        self.leaves_parallel = leaves_parallel
        self.out_file = out_file
        self.special_characters = special_characters
        self.fontname = fontname
        self.rotate = rotate

        # PostScript compatibility for special characters
        if special_characters:
            self.characters = ["&#35;", "<SUB>", "</SUB>", "&le;", "<br/>", ">", "<"]
        else:
            self.characters = ["#", "[", "]", "<=", "\\n", '"', '"']

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {"leaves": []}
        # The colors to render each node with
        self.colors = {"bounds": None}

    def export(self, decision_tree):
        # Check length of feature_names before getting into the tree node
        # Raise error if length of feature_names does not match
        # n_features_in_ in the decision_tree
        if self.feature_names is not None:
            if len(self.feature_names) != decision_tree.n_features_in_:
                raise ValueError(
                    "Length of feature_names, %d does not match number of features, %d"
                    % (len(self.feature_names), decision_tree.n_features_in_)
                )
        # each part writes to out_file
        self.head()
        # Now recurse the tree and add node & edge attributes
        if isinstance(decision_tree, _tree.Tree):
            self.recurse(decision_tree, 0, criterion="impurity")
        else:
            self.recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

        self.tail()

    def tail(self):
        # If required, draw leaf nodes at same depth as each other
        if self.leaves_parallel:
            for rank in sorted(self.ranks):
                self.out_file.write(
                    "{rank=same ; " + "; ".join(r for r in self.ranks[rank]) + "} ;\n"
                )
        self.out_file.write("}")

    def head(self):
        self.out_file.write("digraph Tree {\n")

        # Specify node aesthetics
        self.out_file.write("node [shape=box")
        rounded_filled = []
        if self.filled:
            rounded_filled.append("filled")
        if self.rounded:
            rounded_filled.append("rounded")
        if len(rounded_filled) > 0:
            self.out_file.write(
                ', style="%s", color="black"' % ", ".join(rounded_filled)
            )

        self.out_file.write(', fontname="%s"' % self.fontname)
        self.out_file.write("] ;\n")

        # Specify graph & edge aesthetics
        if self.leaves_parallel:
            self.out_file.write("graph [ranksep=equally, splines=polyline] ;\n")

        self.out_file.write('edge [fontname="%s"] ;\n' % self.fontname)

        if self.rotate:
            self.out_file.write("rankdir=LR ;\n")

    def recurse(self, tree, node_id, criterion, parent=None, depth=0):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Add node with description
        if self.max_depth is None or depth <= self.max_depth:
            # Collect ranks for 'leaf' option in plot_options
            if left_child == _tree.TREE_LEAF:
                self.ranks["leaves"].append(str(node_id))
            elif str(depth) not in self.ranks:
                self.ranks[str(depth)] = [str(node_id)]
            else:
                self.ranks[str(depth)].append(str(node_id))

            self.out_file.write(
                "%d [label=%s" % (node_id, self.node_to_str(tree, node_id, criterion))
            )

            if self.filled:
                self.out_file.write(
                    ', fillcolor="%s"' % self.get_fill_color(tree, node_id)
                )
            self.out_file.write("] ;\n")

            if parent is not None:
                # Add edge to parent
                self.out_file.write("%d -> %d" % (parent, node_id))
                if parent == 0:
                    # Draw True/False labels if parent is root node
                    angles = np.array([45, -45]) * ((self.rotate - 0.5) * -2)
                    self.out_file.write(" [labeldistance=2.5, labelangle=")
                    if node_id == 1:
                        self.out_file.write('%d, headlabel="True"]' % angles[0])
                    else:
                        self.out_file.write('%d, headlabel="False"]' % angles[1])
                self.out_file.write(" ;\n")

            if left_child != _tree.TREE_LEAF:
                self.recurse(
                    tree,
                    left_child,
                    criterion=criterion,
                    parent=node_id,
                    depth=depth + 1,
                )
                self.recurse(
                    tree,
                    right_child,
                    criterion=criterion,
                    parent=node_id,
                    depth=depth + 1,
                )

        else:
            self.ranks["leaves"].append(str(node_id))

            self.out_file.write('%d [label="(...)"' % node_id)
            if self.filled:
                # color cropped nodes grey
                self.out_file.write(', fillcolor="#C0C0C0"')
            self.out_file.write("] ;\n" % node_id)

            if parent is not None:
                # Add edge to parent
                self.out_file.write("%d -> %d ;\n" % (parent, node_id))


class _MPLTreeExporter(_BaseTreeExporter):
    def __init__(
        self,
        max_depth=None,
        feature_names=None,
        class_names=None,
        label="all",
        filled=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        fontsize=None,
        tpt_time_scale=None,
    ):
        super().__init__(
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
        )
        self.fontsize = fontsize

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {"leaves": []}
        # The colors to render each node with
        self.colors = {"bounds": None}

        self.characters = ["#", "[", "]", "<=", "\n", "", ""]
        self.bbox_args = dict()
        if self.rounded:
            self.bbox_args["boxstyle"] = "round"

        self.arrow_args = dict(arrowstyle="<-")

        # TpT-specific attributes (initialized in export method)
        self.is_tpt = False
        self.tree_time_indices = None  # split_time_index array
        self.tree_duration_samples = None  # n_duration_samples array
        self.duration_nodes = {}  # Dict mapping parent_node_id -> duration_node_info
        self.node_parent_times = {}  # Dict mapping node_id -> parent_time (t_p)
        # Dict mapping split_time -> (y, node_id) for axis display
        self.time_axis_nodes = {}
        # Will be set later if None (default: 2.5/mean_split_time)
        self.tpt_time_scale = tpt_time_scale
        self.min_y_distance = 1.5  # Minimum vertical distance between parent and child
        self.time_to_y_mapping = {}  # Dict mapping time_index -> y_position
        # Scale factor for horizontal spacing in TpT trees
        self.tpt_horizontal_spacing_scale = 2.0
        self.tpt_size_scale = 0.5  # Scale factor for node sizes and fonts in TpT

    def _is_tpt_tree(self, decision_tree):
        """Detect TpT trees from the fitted estimator configuration."""
        splitter = getattr(decision_tree, "splitter", None)
        if splitter == "TpT":
            return True

        splitter_cls = getattr(splitter, "__class__", None)
        return getattr(splitter_cls, "__name__", None) == "TpTSplitter"

    def _collect_duration_nodes(self, tree):
        """Collect information about duration nodes (nodes with n_duration_samples > 0).

        Duration values are computed as: parent_value - left_child_value - right_child_value
        This ensures integer counts and consistency with the tree structure.
        """
        for node_id in range(tree.node_count):
            if tree.n_duration_samples[node_id] > 0:
                # This node has a duration branch
                # Compute duration value from parent and children to ensure consistency
                duration_value = self._compute_duration_value_from_children(tree, node_id)

                # Store duration node information
                self.duration_nodes[node_id] = {
                    'n_samples': tree.n_duration_samples[node_id],
                    'weighted_n_samples': tree.weighted_n_duration[node_id],
                    'impurity': tree.impurity_duration[node_id],
                    'value': duration_value,
                    # For proportion calculation
                    'weighted_n_node_samples': tree.weighted_n_duration[node_id],
                }

    def _compute_duration_value_from_children(self, tree, node_id):
        """Compute duration value as parent - left - right to ensure integer counts.

        For each class: duration_value[class] = parent_value[class] - left_value[class] - right_value[class]

        Uses raw counts (n_node_samples) to ensure integer values that sum to N (n_duration_samples).
        """
        # Get parent node value (probabilities stored in tree.value)
        if tree.n_outputs == 1:
            parent_value = tree.value[node_id][0, :].copy()
        else:
            parent_value = tree.value[node_id].copy()

        # Convert parent value to raw counts (not weighted counts)
        # tree.value contains probabilities, multiply by raw count to get raw counts
        if tree.n_classes[0] != 1:  # Classification
            # Use n_node_samples (raw count) to get raw counts per class
            parent_value = parent_value * tree.n_node_samples[node_id]
        else:  # Regression
            # For regression, value is already the mean, keep as is
            pass

        # Get left and right child node IDs
        left_child_id = tree.children_left[node_id]
        right_child_id = tree.children_right[node_id]

        # Initialize duration value as parent value
        duration_value = parent_value.copy()

        # Subtract left child value (using raw counts)
        if left_child_id != _tree.TREE_LEAF:
            if tree.n_outputs == 1:
                left_value = tree.value[left_child_id][0, :].copy()
            else:
                left_value = tree.value[left_child_id].copy()

            # Convert to raw counts if probabilities
            if tree.n_classes[0] != 1:  # Classification
                # Use n_node_samples (raw count) to get raw counts per class
                left_value = left_value * tree.n_node_samples[left_child_id]
            # For regression, value is already the mean

            duration_value -= left_value

        # Subtract right child value (using raw counts)
        if right_child_id != _tree.TREE_LEAF:
            if tree.n_outputs == 1:
                right_value = tree.value[right_child_id][0, :].copy()
            else:
                right_value = tree.value[right_child_id].copy()

            # Convert to raw counts if probabilities
            if tree.n_classes[0] != 1:  # Classification
                # Use n_node_samples (raw count) to get raw counts per class
                right_value = right_value * tree.n_node_samples[right_child_id]
            # For regression, value is already the mean

            duration_value -= right_value

        # Ensure non-negative values (rounding errors might cause small negatives)
        duration_value = np.maximum(duration_value, 0.0)

        # Round to integers for classification (raw counts should be integers)
        if tree.n_classes[0] != 1:  # Classification
            duration_value = np.round(duration_value).astype(int)

        return duration_value

    def _build_parent_time_mapping(self, tree):
        """Build mapping of parent_time (t_p) for each node and identify split times for time axis.

        parent_time is the time when the node was created (parent's split_time_index).
        For root, parent_time = t_0 (minimum time, usually 0).

        For the time axis, we collect all unique parent_time values (excluding duration nodes).
        Time label 0 appears at root's y level.
        Other time labels appear at the y level of children of the least deep node with that parent_time.
        """
        # First, compute parent_time for each node
        def compute_parent_times_recursive(node_id, parent_time=0, depth=0):
            """Recursively compute parent_time for each node."""
            # Store parent_time for this node
            self.node_parent_times[node_id] = parent_time

            # Get this node's split_time_index (time when it splits, or inherited for leaves)
            node_split_time = self.tree_time_indices[node_id]

            # Recurse to children (only left/right children, not duration nodes)
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            if left_child != _tree.TREE_LEAF:
                # Child's parent_time is this node's split_time_index
                compute_parent_times_recursive(left_child, node_split_time, depth + 1)
            if right_child != _tree.TREE_LEAF:
                compute_parent_times_recursive(right_child, node_split_time, depth + 1)

        # Root has parent_time = 0 (t_0)
        compute_parent_times_recursive(0, parent_time=0, depth=0)

        # Collect all unique parent_time values (these are the times that should appear on the axis)
        # Exclude nodes that are duration nodes (they don't have parent_time in the same sense)
        # Duration nodes are identified by having n_duration_samples > 0 in their parent
        parent_times = set()
        for node_id in range(tree.node_count):
            # Skip duration nodes - they don't get time labels
            # Duration nodes are identified by checking if their parent has duration samples
            # Actually, duration nodes are not regular nodes, so we can just collect all parent_times
            parent_time = self.node_parent_times.get(node_id, 0)
            parent_times.add(parent_time)

        # Initialize time_axis_nodes with all parent times (y position will be set later)
        # Time 0 will be at root's y level
        # Other times will be at children's y level of the least deep node with that parent_time
        for parent_time in parent_times:
            self.time_axis_nodes[parent_time] = (None, None)

    def _collect_all_nodes_sorted(self, draw_tree, tree):
        """Collect all nodes sorted by (node_time DESC, depth DESC).

        Returns a list of (node, node_id, node_time, parent_time, depth) tuples.
        node_time = split_time_index (when node splits, or inherited for leaves)
        parent_time = time when node was created (parent's split_time_index)
        """
        all_nodes = []
        self._node_depths = {}

        # Compute depth for each node
        def compute_depths_recursive(node, depth=0):
            """Recursively compute depth for each node."""
            node_id = node.tree.node_id
            self._node_depths[node_id] = depth
            for child in node.children:
                compute_depths_recursive(child, depth + 1)

        compute_depths_recursive(draw_tree, depth=0)

        # Collect all nodes with their metadata
        def collect_nodes_recursive(node):
            """Recursively collect all nodes."""
            node_id = node.tree.node_id
            node_time = self.tree_time_indices[node_id]  # split_time_index
            parent_time = self.node_parent_times.get(node_id, 0)
            depth = self._node_depths[node_id]
            all_nodes.append((node, node_id, node_time, parent_time, depth))
            for child in node.children:
                collect_nodes_recursive(child)

        collect_nodes_recursive(draw_tree)

        # Sort by (node_time DESC, depth DESC) - deepest with latest time first
        # -x[2] = -node_time, -x[4] = -depth
        all_nodes.sort(key=lambda x: (-x[2], -x[4]))

        return all_nodes

    def _adjust_tpt_x_coordinates(self, draw_tree, tree):
        """Adjust x coordinates for TpT trees.

        Starting from deepest/latest nodes:
        1. Nodes at same level (same node_time and depth) should be at least 1.5 units apart
        2. Nodes should be centered above their left and right children
        """
        from collections import defaultdict

        all_nodes = self._collect_all_nodes_sorted(draw_tree, tree)

        # Group nodes by (node_time, depth) for same-level processing
        level_groups = defaultdict(list)
        for node, node_id, node_time, parent_time, depth in all_nodes:
            level_groups[(node_time, depth)].append(node)

        # Process each level group (deepest/latest first)
        for (node_time, depth), nodes_at_level in sorted(
            level_groups.items(), key=lambda x: (-x[0][0], -x[0][1])
        ):
            # Step 1: Ensure minimum spacing of 1.5 between nodes at same level
            nodes_at_level.sort(key=lambda n: n.x)  # Sort by current x position

            for i in range(len(nodes_at_level) - 1):
                current_node = nodes_at_level[i]
                next_node = nodes_at_level[i + 1]
                min_spacing = 1.5
                current_spacing = next_node.x - current_node.x

                if current_spacing < min_spacing:
                    # Shift next_node and all nodes to its right at this level
                    shift_amount = min_spacing - current_spacing
                    next_node.x += shift_amount
                    # Also shift all subsequent nodes at this same level
                    for j in range(i + 2, len(nodes_at_level)):
                        nodes_at_level[j].x += shift_amount

            # Step 2: Center each node above its children
            for node in nodes_at_level:
                if len(node.children) > 0:
                    if len(node.children) >= 2:
                        # Parent at midpoint of leftmost and rightmost children
                        leftmost_x = min(child.x for child in node.children)
                        rightmost_x = max(child.x for child in node.children)
                        node.x = (leftmost_x + rightmost_x) / 2.0
                    elif len(node.children) == 1:
                        # Single child: parent directly above
                        node.x = node.children[0].x

        # Additional pass: ensure all parents are properly centered after spacing adjustments
        # Process from deepest to shallowest (by depth only)
        depth_groups = defaultdict(list)
        for node, node_id, node_time, parent_time, depth in all_nodes:
            depth_groups[depth].append(node)

        for depth in sorted(depth_groups.keys(), reverse=True):  # Deepest first
            for node in depth_groups[depth]:
                if len(node.children) > 0:
                    if len(node.children) >= 2:
                        leftmost_x = min(child.x for child in node.children)
                        rightmost_x = max(child.x for child in node.children)
                        node.x = (leftmost_x + rightmost_x) / 2.0
                    elif len(node.children) == 1:
                        node.x = node.children[0].x

    def _adjust_tpt_y_coordinates(self, draw_tree, tree):
        """Adjust y coordinates for TpT trees based on time differences.

        Starting from deepest/latest nodes:
        - For each node: dt = node_time - parent_node_time
        - parent_node_time is the parent's parent_time (not split_time_index)
        - For root, parent_node_time = 0 (root has no parent)
        - If dt=0, don't change y position
        - If dt≠0, shift this node and all nodes below it down by dt * tpt_time_scale
        - Do NOT change x coordinates during this step
        """
        all_nodes = self._collect_all_nodes_sorted(draw_tree, tree)

        # Build a map to get parent node_id from child node_id
        node_id_to_parent_id = {}
        def build_parent_map(node, parent_id=None):
            """Build mapping from node_id to parent_id."""
            node_id_to_parent_id[node.tree.node_id] = parent_id
            for child in node.children:
                build_parent_map(child, node.tree.node_id)
        build_parent_map(draw_tree, parent_id=None)

        # Process nodes from deepest/latest to shallowest/earliest
        for node, node_id, node_time, parent_time, depth in all_nodes:
            # Calculate dt = node_time - parent_node_time
            # parent_node_time is the parent's parent_time (the time when the parent was created)
            # For root node, parent doesn't exist, so parent_node_time = 0
            # For other nodes, parent_node_time = self.node_parent_times[parent_id]
            parent_id = node_id_to_parent_id.get(node_id)
            if parent_id is not None:
                # This node's parent_node_time is the parent's parent_time
                parent_node_time = self.node_parent_times[parent_id]
            else:
                # Root node: parent doesn't exist, so parent_node_time = 0
                parent_node_time = 0

            # dt = node_time - parent_node_time
            # node_time is the split_time_index (when this node splits)
            # parent_node_time is when the parent was created
            dt = node_time - parent_node_time

            if dt > 0:
                # Shift this node and all descendants down
                shift_amount = dt * self.tpt_time_scale

                # Collect all descendants (nodes below this one in the tree)
                # This includes direct children and all nodes below them
                descendants = []
                def collect_descendants(n):
                    """Collect all descendants of node n."""
                    for child in n.children:
                        descendants.append(child)
                        collect_descendants(child)

                collect_descendants(node)

                # Shift this node and all descendants down
                node.y += shift_amount
                for desc in descendants:
                    desc.y += shift_amount

        # Ensure sibling children (left and right) are at the same y level
        # They have the same parent_time, so they should be at the same y coordinate
        def align_siblings(node):
            """Ensure sibling children are at the same y level."""
            if len(node.children) >= 2:
                # Get the maximum y coordinate among siblings (the one that was shifted most)
                max_y = max(child.y for child in node.children)
                # Set all siblings to the same y level (the maximum)
                for child in node.children:
                    child.y = max_y
            # Recurse to children
            for child in node.children:
                align_siblings(child)

        align_siblings(draw_tree)

        # Update time axis positions
        # Time 0 always appears at root's y level
        root_node = draw_tree
        root_id = root_node.tree.node_id
        if 0 in self.time_axis_nodes:
            self.time_axis_nodes[0] = (root_node.y, root_id)

        # For other times: find the least deep node with that parent_time,
        # and set the time label at the y level of that node (not its children)
        # Build depth map
        node_depths = {}
        def compute_depths(node, depth=0):
            """Compute depth for each node."""
            node_depths[node.tree.node_id] = depth
            for child in node.children:
                compute_depths(child, depth + 1)
        compute_depths(draw_tree, depth=0)

        # For each parent_time (except 0), find least deep node with that parent_time
        for parent_time in self.time_axis_nodes.keys():
            if parent_time == 0:
                continue  # Already handled above

            # Find all nodes with this parent_time, excluding duration nodes
            candidate_nodes = []
            def find_nodes_with_parent_time(node):
                """Find nodes with given parent_time."""
                node_id = node.tree.node_id
                if self.node_parent_times.get(node_id) == parent_time:
                    # Check if this node is not a duration node
                    # Duration nodes are identified by checking if parent has duration samples
                    # Actually, all regular nodes can be candidates
                    if node_id in node_depths:
                        candidate_nodes.append((node_depths[node_id], node))
                for child in node.children:
                    find_nodes_with_parent_time(child)
            find_nodes_with_parent_time(draw_tree)

            if candidate_nodes:
                # Find the least deep node (minimum depth)
                min_depth = min(depth for depth, _ in candidate_nodes)
                least_deep_nodes = [node for depth, node in candidate_nodes if depth == min_depth]

                # Use the average y coordinate of all nodes at the least depth with this parent_time
                # (they should all have the same y since they're at the same depth, but average for precision)
                avg_y = sum(node.y for node in least_deep_nodes) / len(least_deep_nodes)
                representative_node_id = least_deep_nodes[0].tree.node_id

                # Time label appears at the y level of nodes with this parent_time
                # (the nodes with this parent_time are created at this time)
                self.time_axis_nodes[parent_time] = (avg_y, representative_node_id)

    def _draw_time_axis(self, ax, max_x, max_y, axis_x_data=None):
        """Draw time axis on the left side of the tree with arrow pointing down.

        Displays all split times (split_time_index) at the y level of the children created by each split.
        The axis goes from the top to the very bottom of the figure.

        Parameters
        ----------
        axis_x_data : float, optional
            X position in data coordinates for the axis. If None, uses default -2.0.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        # Position axis on the left side in axes fraction coordinates
        # Convert axis_x_data to axes fraction (same as tree nodes use)
        if axis_x_data is None:
            axis_x_data = -2.0  # Default position

        # Convert x position to axes fraction (matching tree node conversion)
        axis_x_frac = (axis_x_data + 0.5) / max_x if max_x > 0 else 0.0

        # Find min and max y positions for axis bounds (from actual node positions)
        y_positions = [y_pos for y_pos, _ in self.time_axis_nodes.values() if y_pos is not None]
        if not y_positions:
            return

        # Axis goes from top (0) to bottom (1.0) in axes fraction coordinates
        axis_y_top_frac = 0.0  # Top of axes
        axis_y_bottom_frac = 1.0  # Bottom of axes

        # Draw vertical axis line on the left (from top to bottom) in axes fraction
        ax.plot([axis_x_frac, axis_x_frac], [axis_y_top_frac, axis_y_bottom_frac],
                'k-', linewidth=1.5, zorder=1, clip_on=False, transform=ax.transAxes)

        # Draw arrow pointing down at the bottom
        # In axes fraction coordinates: y=0 is top, y=1.0 is bottom
        # Arrow should point from near the bottom (higher y value) to the very bottom (y=1.0)
        arrow_length_frac = 0.02  # Fixed arrow length in axes fraction
        arrow_start_y = axis_y_top_frac   # Start at bottom
        arrow_end_y = axis_y_top_frac - arrow_length_frac  # End a bit below bottom
        arrow = FancyArrowPatch(
            (axis_x_frac, arrow_start_y),
            (axis_x_frac, arrow_end_y),
            arrowstyle='->', mutation_scale=15, linewidth=1.5,
            zorder=2, color='black', clip_on=False, transform=ax.transAxes
        )
        ax.add_patch(arrow)

        # Draw time markers and labels for all split times
        # Filter out entries where y_pos is None (will be set during tree traversal)
        for split_time, (y_pos, node_id) in sorted(self.time_axis_nodes.items()):
            if y_pos is None:
                continue  # Skip if y position not yet computed

            # Convert y position to axes fraction (same as tree nodes: (max_y - node.y - 0.5) / max_y)
            y_frac = (max_y - y_pos - 0.5) / max_y if max_y > 0 else 0.0

            # Draw tick mark (toward the tree) in axes fraction
            tick_length_frac = 0.01  # Tick length in axes fraction
            ax.plot([axis_x_frac, axis_x_frac + tick_length_frac], [y_frac, y_frac],
                   'k-', linewidth=1, zorder=2, clip_on=False, transform=ax.transAxes)

            # Draw time label (to the left of axis) in axes fraction
            tpt_scale = self.tpt_size_scale if self.is_tpt else 1.0
            fontsize_val = (self.fontsize or 10) * tpt_scale
            label_x_frac = axis_x_frac - 0.01  # Small offset to the left of axis
            ax.text(label_x_frac, y_frac, str(split_time),
                   ha='right', va='center',
                   fontsize=fontsize_val, zorder=3, clip_on=False, transform=ax.transAxes)

    def _draw_duration_node(self, parent_node, tree, ax, max_x, max_y, depth, criterion):
        """Draw a duration node to the right of its parent node.

        Duration nodes are displayed with the same information as leaf nodes:
        - Impurity (criterion value)
        - Number of samples
        - Target value
        """
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation

        duration_info = self.duration_nodes[parent_node.tree.node_id]

        # Position duration node to the right of parent
        duration_x_offset = 1  # Horizontal offset from parent
        duration_x = parent_node.x - duration_x_offset
        duration_y = parent_node.y  # Same y level as parent

        # Convert to axes fraction
        xy_duration = ((duration_x + 0.5) / max_x, (max_y - duration_y - 0.5) / max_y)

        # Generate label for duration node (similar to leaf node format)
        duration_label = self._format_duration_node_label(duration_info, tree, criterion)

        # Draw duration node annotation
        kwargs = dict(
            ha="center",
            va="center",
            bbox=self.bbox_args.copy(),
            xycoords="axes fraction",
            zorder=100 - 10 * depth,
        )
        # Apply TpT size scaling to fontsize
        if self.fontsize is not None:
            fontsize_val = self.fontsize * (self.tpt_size_scale if self.is_tpt else 1.0)
            kwargs["fontsize"] = fontsize_val
        elif self.is_tpt:
            kwargs["fontsize"] = 10 * self.tpt_size_scale

        # Style duration nodes differently (e.g., dashed border or different color)
        if self.filled:
            kwargs["bbox"]["fc"] = self.get_fill_color(tree, parent_node.tree.node_id)
        else:
            kwargs["bbox"]["fc"] = ax.get_facecolor()

        # Add visual distinction for duration nodes (dashed border)
        kwargs["bbox"]["linestyle"] = "--"

        ax.annotate(duration_label, xy_duration, **kwargs)

        # Draw connecting line from parent to duration node
        xy_parent = ((parent_node.x + 0.5) / max_x, (max_y - parent_node.y - 0.5) / max_y)
        ax.plot([xy_parent[0], xy_duration[0]], [xy_parent[1], xy_duration[1]],
               'k--', linewidth=1, transform=ax.transAxes, zorder=50)

    def _format_duration_node_label(self, duration_info, tree, criterion):
        """Format the label string for a duration node (exactly like leaf format)."""
        label_parts = []
        characters = self.characters

        # Add "Duration" identifier
        label_parts.append("Duration" + characters[4])

        # Labels are always shown for duration nodes (similar to "all" setting)
        labels = True

        # Add impurity (same format as regular nodes)
        if self.impurity:
            if isinstance(criterion, _criterion.FriedmanMSE):
                criterion_name = "friedman_mse"
            elif isinstance(criterion, _criterion.MSE) or criterion == "squared_error":
                criterion_name = "squared_error"
            elif not isinstance(criterion, str):
                criterion_name = "impurity"
            else:
                criterion_name = criterion

            if duration_info['impurity'] == float('inf') or np.isinf(duration_info['impurity']):
                imp_str = "inf"
            else:
                imp_str = str(round(duration_info['impurity'], self.precision))

            if labels:
                label_parts.append("%s = " % criterion_name)
            label_parts.append(imp_str + characters[4])

        # Add samples (same format as regular nodes)
        if labels:
            label_parts.append("samples = ")
        if self.proportion:
            # Calculate percentage based on root node samples
            percent = 100.0 * duration_info['n_samples'] / float(tree.n_node_samples[0])
            label_parts.append(str(round(percent, 1)) + "%" + characters[4])
        else:
            label_parts.append(str(duration_info['n_samples']) + characters[4])

        # Add value (same format as regular leaf nodes)
        # duration_info['value'] already contains raw counts computed from parent - left - right
        if duration_info['value'] is not None:
            value = duration_info['value'].copy()

            # No need to multiply - value already contains raw counts (integers)
            # that sum to n_samples

            if labels:
                label_parts.append("value = ")

            if tree.n_classes[0] == 1:
                # Regression
                value_text = np.around(value, self.precision)
            elif self.proportion:
                # Classification
                value_text = np.around(value, self.precision)
            elif np.all(np.equal(np.mod(value, 1), 0)):
                # Classification without floating-point weights
                value_text = value.astype(int)
            else:
                # Classification with floating-point weights
                value_text = np.around(value, self.precision)

            # Strip whitespace (same as regular nodes)
            value_str = str(value_text.astype("S32")).replace("b'", "'")
            value_str = value_str.replace("' '", ", ").replace("'", "")
            if tree.n_classes[0] == 1 and tree.n_outputs == 1:
                value_str = value_str.replace("[", "").replace("]", "")
            value_str = value_str.replace("\n ", characters[4])
            label_parts.append(value_str + characters[4])

        # Write node majority class (same as regular nodes)
        if (
            self.class_names is not None
            and tree.n_classes[0] != 1
            and tree.n_outputs == 1
            and duration_info['value'] is not None
        ):
            # Only done for single-output classification trees
            if labels:
                label_parts.append("class = ")
            if self.class_names is not True:
                class_name = self.class_names[np.argmax(duration_info['value'])]
            else:
                class_name = "y%s%s%s" % (
                    characters[1],
                    np.argmax(duration_info['value']),
                    characters[2],
                )
            label_parts.append(class_name)

        # Join all parts
        label = "".join(label_parts)
        if label.endswith(characters[4]):
            label = label[:-len(characters[4])]

        return label + characters[5]

    def _make_tree(self, node_id, et, criterion, depth=0):
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        name = self.node_to_str(et, node_id, criterion=criterion)
        if et.children_left[node_id] != _tree.TREE_LEAF and (
            self.max_depth is None or depth <= self.max_depth
        ):
            children = [
                self._make_tree(
                    et.children_left[node_id], et, criterion, depth=depth + 1
                ),
                self._make_tree(
                    et.children_right[node_id], et, criterion, depth=depth + 1
                ),
            ]
        else:
            return Tree(name, node_id)
        return Tree(name, node_id, *children)

    def export(self, decision_tree, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()

        # Detect if this is a TpT tree and initialize TpT-specific attributes
        tree_ = decision_tree.tree_
        self.is_tpt = self._is_tpt_tree(decision_tree)
        if self.is_tpt:
            self.tree_time_indices = tree_.split_time_index
            self.tree_duration_samples = tree_.n_duration_samples
            self.duration_nodes = {}
            # Collect duration nodes information
            self._collect_duration_nodes(tree_)
            # Build parent_time mapping for each node and identify time axis nodes
            self._build_parent_time_mapping(tree_)

            # Compute default tpt_time_scale if not provided
            if self.tpt_time_scale is None:
                # Default: 2.5 / mean_split_time, where mean_split_time is the mean of all non-leaf nodes' split times
                split_times = []
                for node_id in range(tree_.node_count):
                    left_child = tree_.children_left[node_id]
                    if left_child != _tree.TREE_LEAF:  # Non-leaf node
                        split_time = self.tree_time_indices[node_id]
                        split_times.append(split_time)
                if len(split_times) > 0:
                    mean_split_time = np.mean(split_times)
                    self.tpt_time_scale = 2.5 / mean_split_time
                else:
                    self.tpt_time_scale = 0.5  # Fallback if no non-leaf nodes

        my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
        draw_tree = buchheim(my_tree)

        # Adjust coordinates for TpT
        # Order: first x coordinates, then y coordinates (as specified)
        if self.is_tpt:
            self._adjust_tpt_x_coordinates(draw_tree, tree_)  # First adjust x coordinates
            self._adjust_tpt_y_coordinates(draw_tree, tree_)  # Then adjust y coordinates

        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        max_x, max_y = draw_tree.max_extents() + 1

        # For TpT trees, account for duration nodes in horizontal spacing
        if self.is_tpt and len(self.duration_nodes) > 0:
            # Duration nodes are positioned 1.5 units to the right of their parent
            # Add extra horizontal space to accommodate them
            max_x += 2.0  # Add extra space for duration nodes

        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y

        # Draw tree nodes first
        self.recurse(draw_tree, decision_tree.tree_, ax, max_x, max_y, criterion=decision_tree.criterion)

        # Draw time axis for TpT trees (after drawing tree nodes, so we know final positions)
        if self.is_tpt and len(self.time_axis_nodes) > 0:
            # Find minimum x coordinate after all adjustments
            min_x = float('inf')
            def find_min_x(node):
                nonlocal min_x
                if node.x < min_x:
                    min_x = node.x
                for child in node.children:
                    find_min_x(child)
            find_min_x(draw_tree)
            # Position time axis at the very left of the figure
            # We want it at x_frac = 0.0 (left edge) in axes fraction coordinates
            # Since tree nodes use: x_frac = (x_data + 0.5) / max_x
            # To get x_frac = 0.0, we need: x_data = -0.5
            # But we also want a small margin, so use a small negative value
            time_axis_x = -0.5  # This positions axis at x_frac ≈ 0.0 (very left)
            self._draw_time_axis(ax, max_x, max_y, time_axis_x)

        anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]

        # update sizes of all bboxes
        renderer = ax.figure.canvas.get_renderer()

        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            # get figure to data transform
            # adjust fontsize to avoid overlap
            # get max box width and height
            extents = [
                bbox_patch.get_window_extent()
                for ann in anns
                if (bbox_patch := ann.get_bbox_patch()) is not None
            ]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            # width should be around scale_x in axis coordinates
            base_fontsize = anns[0].get_fontsize()
            # Apply TpT size scaling if needed
            if self.is_tpt:
                base_fontsize = base_fontsize / self.tpt_size_scale  # Unscale to get original, then apply again
            size = base_fontsize * min(
                scale_x / max_width, scale_y / max_height
            )
            if self.is_tpt:
                size = size * self.tpt_size_scale  # Apply TpT scale
            for ann in anns:
                ann.set_fontsize(size)

        return anns

    def recurse(self, node, tree, ax, max_x, max_y, depth=0, criterion=None):
        import matplotlib.pyplot as plt

        # kwargs for annotations without a bounding box
        common_kwargs = dict(
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
        )
        # Apply TpT size scaling to fontsize
        if self.fontsize is not None:
            fontsize_val = self.fontsize * (self.tpt_size_scale if self.is_tpt else 1.0)
            common_kwargs["fontsize"] = fontsize_val
        elif self.is_tpt:
            # Even if fontsize is None, apply scale for TpT
            common_kwargs["fontsize"] = 10 * self.tpt_size_scale

        # kwargs for annotations with a bounding box
        bbox_args_copy = self.bbox_args.copy()
        # Scale down bbox padding for TpT
        if self.is_tpt:
            if 'boxstyle' in bbox_args_copy:
                # Adjust boxstyle parameters
                if isinstance(bbox_args_copy['boxstyle'], str):
                    # Simple string boxstyle - add padding scale
                    pass
            if 'pad' in bbox_args_copy:
                bbox_args_copy['pad'] = bbox_args_copy['pad'] * self.tpt_size_scale

        kwargs = dict(
            ha="center",
            va="center",
            bbox=bbox_args_copy,
            arrowprops=self.arrow_args.copy(),
            **common_kwargs,
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        # offset things by .5 to center them in plot
        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs["bbox"]["fc"] = self.get_fill_color(tree, node.tree.node_id)
            else:
                kwargs["bbox"]["fc"] = ax.get_facecolor()

            if node.parent is None:
                # root
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.5) / max_y,
                )
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)

                # Draw True/False labels if parent is root node
                if node.parent.parent is None:
                    # Adjust the position for the text to be slightly above the arrow
                    text_pos = (
                        (xy_parent[0] + xy[0]) / 2,
                        (xy_parent[1] + xy[1]) / 2,
                    )
                    # Annotate the arrow with the edge label to indicate the child
                    # where the sample-split condition is satisfied
                    if node.parent.left() == node:
                        label_text, label_ha = ("True  ", "right")
                    else:
                        label_text, label_ha = ("  False", "left")
                    ax.annotate(label_text, text_pos, ha=label_ha, **common_kwargs)
            # Draw duration node if this parent has one (TpT only)
            if self.is_tpt and node.tree.node_id in self.duration_nodes:
                self._draw_duration_node(node, tree, ax, max_x, max_y, depth, criterion)

            for child in node.children:
                self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1, criterion=criterion)

        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "grey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)


@validate_params(
    {
        "decision_tree": "no_validation",
        "out_file": [str, None, HasMethods("write")],
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],
        "feature_names": ["array-like", None],
        "class_names": ["array-like", "boolean", None],
        "label": [StrOptions({"all", "root", "none"})],
        "filled": ["boolean"],
        "leaves_parallel": ["boolean"],
        "impurity": ["boolean"],
        "node_ids": ["boolean"],
        "proportion": ["boolean"],
        "rotate": ["boolean"],
        "rounded": ["boolean"],
        "special_characters": ["boolean"],
        "precision": [Interval(Integral, 0, None, closed="left"), None],
        "fontname": [str],
    },
    prefer_skip_nested_validation=True,
)
def export_graphviz(
    decision_tree,
    out_file=None,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    filled=False,
    leaves_parallel=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rotate=False,
    rounded=False,
    special_characters=False,
    precision=3,
    fontname="helvetica",
):
    """Export a decision tree in DOT format.

    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    decision_tree : object
        The decision tree estimator to be exported to GraphViz.

    out_file : object or str, default=None
        Handle or name of the output file. If ``None``, the result is
        returned as a string.

        .. versionchanged:: 0.20
            Default of out_file changed from "tree.dot" to None.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : array-like of shape (n_features,), default=None
        An array containing the feature names.
        If None, generic names will be used ("x[0]", "x[1]", ...).

    class_names : array-like of shape (n_classes,) or bool, default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, default=False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    leaves_parallel : bool, default=False
        When set to ``True``, draw all leaf nodes at the bottom of the tree.

    impurity : bool, default=True
        When set to ``True``, show the impurity at each node.

    node_ids : bool, default=False
        When set to ``True``, show the ID number on each node.

    proportion : bool, default=False
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rotate : bool, default=False
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, default=False
        When set to ``True``, draw node boxes with rounded corners.

    special_characters : bool, default=False
        When set to ``False``, ignore special characters for PostScript
        compatibility.

    precision : int, default=3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    fontname : str, default='helvetica'
        Name of font used to render text.

    Returns
    -------
    dot_data : str
        String representation of the input tree in GraphViz dot format.
        Only returned if ``out_file`` is None.

        .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier()
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.export_graphviz(clf)
    'digraph Tree {...
    """
    if feature_names is not None:
        feature_names = check_array(
            feature_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )
    if class_names is not None and not isinstance(class_names, bool):
        class_names = check_array(
            class_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )

    check_is_fitted(decision_tree)
    own_file = False
    return_string = False
    try:
        if isinstance(out_file, str):
            out_file = open(out_file, "w", encoding="utf-8")
            own_file = True

        if out_file is None:
            return_string = True
            out_file = StringIO()

        exporter = _DOTTreeExporter(
            out_file=out_file,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            leaves_parallel=leaves_parallel,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rotate=rotate,
            rounded=rounded,
            special_characters=special_characters,
            precision=precision,
            fontname=fontname,
        )
        exporter.export(decision_tree)

        if return_string:
            return exporter.out_file.getvalue()

    finally:
        if own_file:
            out_file.close()


def _compute_depth(tree, node):
    """
    Returns the depth of the subtree rooted in node.
    """

    def compute_depth_(
        current_node, current_depth, children_left, children_right, depths
    ):
        depths += [current_depth]
        left = children_left[current_node]
        right = children_right[current_node]
        if left != -1 and right != -1:
            compute_depth_(
                left, current_depth + 1, children_left, children_right, depths
            )
            compute_depth_(
                right, current_depth + 1, children_left, children_right, depths
            )

    depths = []
    compute_depth_(node, 1, tree.children_left, tree.children_right, depths)
    return max(depths)


@validate_params(
    {
        "decision_tree": [DecisionTreeClassifier, DecisionTreeRegressor],
        "feature_names": ["array-like", None],
        "class_names": ["array-like", None],
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],
        "spacing": [Interval(Integral, 1, None, closed="left"), None],
        "decimals": [Interval(Integral, 0, None, closed="left"), None],
        "show_weights": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def export_text(
    decision_tree,
    *,
    feature_names=None,
    class_names=None,
    max_depth=10,
    spacing=3,
    decimals=2,
    show_weights=False,
):
    """Build a text report showing the rules of a decision tree.

    Note that backwards compatibility may not be supported.

    Parameters
    ----------
    decision_tree : object
        The decision tree estimator to be exported.
        It can be an instance of
        DecisionTreeClassifier or DecisionTreeRegressor.

    feature_names : array-like of shape (n_features,), default=None
        An array containing the feature names.
        If None generic names will be used ("feature_0", "feature_1", ...).

    class_names : array-like of shape (n_classes,), default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.

        - if `None`, the class names are delegated to `decision_tree.classes_`;
        - otherwise, `class_names` will be used as class names instead of
          `decision_tree.classes_`. The length of `class_names` must match
          the length of `decision_tree.classes_`.

        .. versionadded:: 1.3

    max_depth : int, default=10
        Only the first max_depth levels of the tree are exported.
        Truncated branches will be marked with "...".

    spacing : int, default=3
        Number of spaces between edges. The higher it is, the wider the result.

    decimals : int, default=2
        Number of decimal digits to display.

    show_weights : bool, default=False
        If true the classification weights will be exported on each leaf.
        The classification weights are the number of samples each class.

    Returns
    -------
    report : str
        Text summary of all the rules in the decision tree.

    Examples
    --------

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.tree import export_text
    >>> iris = load_iris()
    >>> X = iris['data']
    >>> y = iris['target']
    >>> decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    >>> decision_tree = decision_tree.fit(X, y)
    >>> r = export_text(decision_tree, feature_names=iris['feature_names'])
    >>> print(r)
    |--- petal width (cm) <= 0.80
    |   |--- class: 0
    |--- petal width (cm) >  0.80
    |   |--- petal width (cm) <= 1.75
    |   |   |--- class: 1
    |   |--- petal width (cm) >  1.75
    |   |   |--- class: 2
    """
    if feature_names is not None:
        feature_names = check_array(
            feature_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )
    if class_names is not None:
        class_names = check_array(
            class_names, ensure_2d=False, dtype=None, ensure_min_samples=0
        )

    check_is_fitted(decision_tree)
    tree_ = decision_tree.tree_
    if is_classifier(decision_tree):
        if class_names is None:
            class_names = decision_tree.classes_
        elif len(class_names) != len(decision_tree.classes_):
            raise ValueError(
                "When `class_names` is an array, it should contain as"
                " many items as `decision_tree.classes_`. Got"
                f" {len(class_names)} while the tree was fitted with"
                f" {len(decision_tree.classes_)} classes."
            )
    right_child_fmt = "{} {} <= {}\n"
    left_child_fmt = "{} {} >  {}\n"
    truncation_fmt = "{} {}\n"

    if feature_names is not None and len(feature_names) != tree_.n_features:
        raise ValueError(
            "feature_names must contain %d elements, got %d"
            % (tree_.n_features, len(feature_names))
        )

    if isinstance(decision_tree, DecisionTreeClassifier):
        value_fmt = "{}{} weights: {}\n"
        if not show_weights:
            value_fmt = "{}{}{}\n"
    else:
        value_fmt = "{}{} value: {}\n"

    if feature_names is not None:
        feature_names_ = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else None
            for i in tree_.feature
        ]
    else:
        feature_names_ = ["feature_{}".format(i) for i in tree_.feature]

    export_text.report = ""

    def _add_leaf(value, weighted_n_node_samples, class_name, indent):
        val = ""
        if isinstance(decision_tree, DecisionTreeClassifier):
            if show_weights:
                val = [
                    "{1:.{0}f}, ".format(decimals, v * weighted_n_node_samples)
                    for v in value
                ]
                val = "[" + "".join(val)[:-2] + "]"
                weighted_n_node_samples
            val += " class: " + str(class_name)
        else:
            val = ["{1:.{0}f}, ".format(decimals, v) for v in value]
            val = "[" + "".join(val)[:-2] + "]"
        export_text.report += value_fmt.format(indent, "", val)

    def print_tree_recurse(node, depth):
        indent = ("|" + (" " * spacing)) * depth
        indent = indent[:-spacing] + "-" * spacing

        value = None
        if tree_.n_outputs == 1:
            value = tree_.value[node][0]
        else:
            value = tree_.value[node].T[0]
        class_name = np.argmax(value)

        if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:
            class_name = class_names[class_name]

        weighted_n_node_samples = tree_.weighted_n_node_samples[node]

        if depth <= max_depth + 1:
            info_fmt = ""
            info_fmt_left = info_fmt
            info_fmt_right = info_fmt

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names_[node]
                threshold = tree_.threshold[node]
                threshold = "{1:.{0}f}".format(decimals, threshold)
                export_text.report += right_child_fmt.format(indent, name, threshold)
                export_text.report += info_fmt_left
                print_tree_recurse(tree_.children_left[node], depth + 1)

                export_text.report += left_child_fmt.format(indent, name, threshold)
                export_text.report += info_fmt_right
                print_tree_recurse(tree_.children_right[node], depth + 1)
            else:  # leaf
                _add_leaf(value, weighted_n_node_samples, class_name, indent)
        else:
            subtree_depth = _compute_depth(tree_, node)
            if subtree_depth == 1:
                _add_leaf(value, weighted_n_node_samples, class_name, indent)
            else:
                trunc_report = "truncated branch of depth %d" % subtree_depth
                export_text.report += truncation_fmt.format(indent, trunc_report)

    print_tree_recurse(0, 1)
    return export_text.report

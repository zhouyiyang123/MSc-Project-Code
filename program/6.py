import math
import tkinter as tk
from bisect import bisect_left
from collections import defaultdict
from tkinter import messagebox

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path


def build_edge_compat_index(D):
    # Precompute, for each ordered pair (v, w), the interval structure used by
    # edge compatibility checks. This turns repeated scans into fast lookups.
    n = len(D)
    compat_index = {}

    for v in range(n):
        for w in range(n):
            intervals = []

            for x in range(n):
                left = D[x][v]
                right = D[x][w]
                if left < math.inf and left < right:
                    intervals.append((left, right))

            intervals.sort(key=lambda item: item[0])

            starts = []
            prefix_max_right = []
            current_max_right = -math.inf

            for left, right in intervals:
                starts.append(left)
                if right > current_max_right:
                    current_max_right = right
                prefix_max_right.append(current_max_right)

            compat_index[(v, w)] = (starts, prefix_max_right)

    return compat_index


def edge_compat_fast(compat_index, v, w, t):
    # Return False iff t lies inside any open interval (D[x][v], D[x][w]).
    starts, prefix_max_right = compat_index[(v, w)]
    interval_index = bisect_left(starts, t) - 1

    if interval_index < 0:
        return True

    return prefix_max_right[interval_index] <= t


def foremost_realization_from_matrix_optimized(D):
    n = len(D)
    couples = []

    for u in range(n):
        for w in range(n):
            if u != w and D[u][w] < math.inf:
                couples.append((u, w))

    compat_index = build_edge_compat_index(D)
    lambda_edges = defaultdict(list)

    for u, w in couples:
        target_time = D[u][w]
        found = False

        for v in range(n):
            if D[u][v] < target_time and edge_compat_fast(compat_index, v, w, target_time):
                lambda_edges[(v, w)].append(target_time)
                found = True
                break

        if not found:
            return False, None

    return True, lambda_edges


def build_directed_temporal_graph(lambda_edges, n):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))

    for (u, v), times in lambda_edges.items():
        graph.add_edge(u, v, times=sorted(times))

    return graph


def build_underlying_graph(lambda_edges, n):
    # The underlying graph ignores edge directions and time labels.
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for (u, v) in lambda_edges:
        if u != v:
            graph.add_edge(u, v)

    return graph


def test_underlying_planarity(lambda_edges, n):
    underlying_graph = build_underlying_graph(lambda_edges, n)
    is_planar = is_planar_by_kuratowski(underlying_graph)
    return is_planar, underlying_graph


def _simple_graph_copy(graph):
    # Convert the graph into a simple undirected graph:
    # remove direction effects, duplicate edges, and self-loops.
    simple_graph = nx.Graph()
    simple_graph.add_nodes_from(graph.nodes())
    simple_graph.add_edges_from(graph.edges())
    simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))
    return simple_graph


def _trim_isolates(graph):
    # Remove isolated vertices because they cannot help form a Kuratowski minor.
    isolated_nodes = [node for node, degree in graph.degree() if degree == 0]
    if isolated_nodes:
        graph.remove_nodes_from(isolated_nodes)


def _graph_state_key(graph):
    # Build a compact signature so we can avoid revisiting the same subproblem.
    ordered_nodes = sorted(graph.nodes(), key=lambda node: (graph.degree(node), str(node)))
    relabeled = nx.convert_node_labels_to_integers(graph.subgraph(ordered_nodes).copy())
    edges = tuple(sorted((min(u, v), max(u, v)) for u, v in relabeled.edges()))
    return relabeled.number_of_nodes(), edges


def _is_target_subgraph(graph, target):
    # Check whether the current graph already contains the target graph directly.
    if graph.number_of_nodes() < target.number_of_nodes():
        return False
    matcher = nx.algorithms.isomorphism.GraphMatcher(graph, target)
    return matcher.subgraph_is_isomorphic()


def _contract_edge(graph, u, v):
    # Contract edge (u, v): merge v into u and reconnect all neighbors of v to u.
    contracted = graph.copy()
    if not contracted.has_edge(u, v):
        return contracted

    neighbors = set(contracted.neighbors(u)) | set(contracted.neighbors(v))
    contracted.remove_node(v)

    for neighbor in neighbors:
        if neighbor != u:
            contracted.add_edge(u, neighbor)

    contracted.remove_edges_from(nx.selfloop_edges(contracted))
    _trim_isolates(contracted)
    return contracted


def _choose_branch_edge(graph):
    # Pick a promising edge to branch on first.
    # We prefer edges with larger endpoint degrees to reach dense structures sooner.
    best_edge = None
    best_score = -1

    for u, v in graph.edges():
        score = graph.degree(u) + graph.degree(v)
        if score > best_score:
            best_score = score
            best_edge = (u, v)

    return best_edge


def _contains_minor_recursive(graph, target, seen):
    # Recursive minor search:
    # try to find the target graph by repeatedly deleting or contracting edges.
    graph = _simple_graph_copy(graph)
    _trim_isolates(graph)

    if graph.number_of_nodes() < target.number_of_nodes():
        return False

    if graph.number_of_edges() < target.number_of_edges():
        return False

    key = _graph_state_key(graph)
    if key in seen:
        return False
    seen.add(key)

    if _is_target_subgraph(graph, target):
        return True

    branch_edge = _choose_branch_edge(graph)
    if branch_edge is None:
        return False

    u, v = branch_edge

    deleted = graph.copy()
    deleted.remove_edge(u, v)
    _trim_isolates(deleted)
    if _contains_minor_recursive(deleted, target, seen):
        return True

    contracted = _contract_edge(graph, u, v)
    if _contains_minor_recursive(contracted, target, seen):
        return True

    return False


def contains_kuratowski_minor(graph):
    # Kuratowski's theorem:
    # a graph is non-planar iff it contains K5 or K3,3 as a Kuratowski obstruction.
    k5 = nx.complete_graph(5)
    k33 = nx.complete_bipartite_graph(3, 3)
    return _contains_minor_recursive(graph, k5, set()) or _contains_minor_recursive(
        graph, k33, set()
    )


def is_planar_by_kuratowski(graph):
    # Main planarity test used by the program.
    # First apply a quick edge-count filter, then search for K5/K3,3 minors.
    simple_graph = _simple_graph_copy(graph)

    for component_nodes in nx.connected_components(simple_graph):
        component = simple_graph.subgraph(component_nodes).copy()
        if component.number_of_nodes() <= 4:
            continue

        if component.number_of_edges() > 3 * component.number_of_nodes() - 6:
            return False

        if contains_kuratowski_minor(component):
            return False

    return True


def draw_underlying_graph(underlying_graph, is_planar):
    # Show the underlying graph itself so the planarity result is easier to see.
    if is_planar and hasattr(nx, "planar_layout"):
        try:
            pos = nx.planar_layout(underlying_graph)
            layout_name = "Planar layout"
        except Exception:
            pos = nx.spring_layout(underlying_graph, seed=42)
            layout_name = "Spring layout"
    else:
        pos = nx.spring_layout(underlying_graph, seed=42)
        layout_name = "Spring layout"

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    nx.draw_networkx_nodes(
        underlying_graph,
        pos,
        ax=ax,
        node_size=900,
        node_color="#111111",
    )
    nx.draw_networkx_edges(
        underlying_graph,
        pos,
        ax=ax,
        width=1.8,
        edge_color="#333333",
    )
    nx.draw_networkx_labels(
        underlying_graph,
        pos,
        ax=ax,
        font_size=13,
        font_color="white",
        font_weight="bold",
    )

    status_text = "Planar" if is_planar else "Not planar"
    title_color = "#1b8f3a" if is_planar else "#c62828"
    ax.set_title(
        f"Underlying Graph ({layout_name}) - {status_text}",
        fontsize=15,
        color=title_color,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_temporal_graph(lambda_edges, n):
    curve_offset_ratio = 0.32
    node_size = 80
    node_radius_points = math.sqrt(node_size / math.pi)
    arrow_shrink = node_radius_points * 0.7
    label_offset_x = 0.03

    graph = build_directed_temporal_graph(lambda_edges, n)
    pos = nx.spring_layout(graph, seed=42)
    fig, ax = plt.subplots(figsize=(8, 8))

    def quadratic_bezier_point(start, control, end, t=0.5):
        x = (
            (1 - t) * (1 - t) * start[0]
            + 2 * (1 - t) * t * control[0]
            + t * t * end[0]
        )
        y = (
            (1 - t) * (1 - t) * start[1]
            + 2 * (1 - t) * t * control[1]
            + t * t * end[1]
        )
        return x, y

    def build_edge_geometry(small, large, direction, side):
        small_x, small_y = pos[small]
        large_x, large_y = pos[large]
        dx = large_x - small_x
        dy = large_y - small_y
        length = math.hypot(dx, dy)
        if length == 0:
            point = (small_x, small_y)
            return point, point, point

        unit_x = dx / length
        unit_y = dy / length
        normal_x = -unit_y
        normal_y = unit_x
        offset = curve_offset_ratio * length

        if direction == "up":
            start = (small_x, small_y)
            end = (large_x, large_y)
        else:
            start = (large_x, large_y)
            end = (small_x, small_y)

        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        control = (mid_x + normal_x * offset * side, mid_y + normal_y * offset * side)
        return start, control, end

    processed_pairs = set()
    for u, v in graph.edges():
        pair = frozenset((u, v))
        if pair in processed_pairs:
            continue

        reverse_exists = graph.has_edge(v, u)
        if reverse_exists:
            small, large = sorted((u, v))
            edge_specs = [
                (small, large, "up", 1),
                (large, small, "down", -1),
            ]
        else:
            edge_specs = [(u, v, "single", 0)]

        for start_node, end_node, direction, side in edge_specs:
            edge_color = "black"

            if direction == "single":
                x1, y1 = pos[start_node]
                x2, y2 = pos[end_node]
                start = (x1, y1)
                end = (x2, y2)
                control = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
                label_x, label_y = quadratic_bezier_point(start, control, end, t=0.5)
            else:
                small, large = sorted((start_node, end_node))
                start, control, end = build_edge_geometry(small, large, direction, side)
                label_x, label_y = quadratic_bezier_point(start, control, end, t=0.5)

            arrow_path = Path(
                [start, control, end],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3],
            )

            arrow = FancyArrowPatch(
                path=arrow_path,
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=1.8,
                color=edge_color,
                shrinkA=arrow_shrink,
                shrinkB=arrow_shrink,
                zorder=1,
            )
            ax.add_patch(arrow)

            label = f"t={', '.join(map(str, graph[start_node][end_node]['times']))}"
            ax.text(
                label_x,
                label_y,
                label,
                fontsize=11,
                ha="center",
                va="center",
                color=edge_color,
                zorder=3,
            )

        processed_pairs.add(pair)

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_size,
        node_color="black",
    )

    for node, (x, y) in pos.items():
        ax.text(
            x + label_offset_x,
            y,
            str(node),
            fontsize=14,
            fontweight="bold",
            color="black",
            ha="left",
            va="center",
            zorder=4,
        )

    ax.set_title("Temporal Graph with Time Labels", fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def parse_matrix_value(text):
    value = text.strip().lower()
    if value in {"inf", "+inf", "infinity"}:
        return math.inf

    number = float(text)
    if number.is_integer():
        return int(number)
    return number


def parse_matrix_input(size_text, matrix_text):
    try:
        n = int(size_text.strip())
    except ValueError as exc:
        raise ValueError("Matrix size n must be an integer.") from exc

    if n <= 0:
        raise ValueError("Matrix size n must be greater than 0.")

    lines = [line.strip() for line in matrix_text.splitlines() if line.strip()]
    if len(lines) != n:
        raise ValueError(f"You must enter exactly {n} rows. Current rows: {len(lines)}.")

    matrix = []
    for row_index, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != n:
            raise ValueError(f"Row {row_index} must contain exactly {n} values.")

        try:
            row = [parse_matrix_value(part) for part in parts]
        except ValueError as exc:
            raise ValueError(
                f"Row {row_index} contains an invalid value. Use numbers or inf only."
            ) from exc

        matrix.append(row)

    return matrix


def get_user_configuration():
    result = {"matrix": None, "mode": "visualize"}

    root = tk.Tk()
    root.title("Temporal Graph Matrix Input")
    root.geometry("760x600")
    root.minsize(760, 600)

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)

    # Keep the action button visible by separating the content area from the bottom bar.
    content_frame = tk.Frame(main_frame)
    content_frame.pack(fill=tk.BOTH, expand=True)

    title_label = tk.Label(
        content_frame,
        text="Enter the matrix, choose a mode, and click Confirm",
        font=("Microsoft YaHei", 13, "bold"),
    )
    title_label.pack(pady=(0, 6), anchor="w")

    size_frame = tk.Frame(content_frame)
    size_frame.pack(pady=6, anchor="w")

    size_label = tk.Label(size_frame, text="Matrix size n:")
    size_label.pack(side=tk.LEFT, padx=(0, 8))

    size_entry = tk.Entry(size_frame, width=10)
    size_entry.pack(side=tk.LEFT)

    mode_frame = tk.LabelFrame(content_frame, text="Output mode", padx=10, pady=8)
    mode_frame.pack(fill=tk.X, pady=(8, 6))

    mode_var = tk.StringVar(value="visualize")

    visualize_radio = tk.Radiobutton(
        mode_frame,
        text="Visualize the temporal graph",
        variable=mode_var,
        value="visualize",
        anchor="w",
        justify="left",
    )
    visualize_radio.pack(fill=tk.X, anchor="w")

    planar_radio = tk.Radiobutton(
        mode_frame,
        text="Test whether the underlying graph is planar",
        variable=mode_var,
        value="planarity",
        anchor="w",
        justify="left",
    )
    planar_radio.pack(fill=tk.X, anchor="w")

    hint_text = (
        "Enter the matrix row by row, with spaces between values; use inf for unreachable.\n"
        "Example:\n"
        "0 1 2 3\n"
        "inf 0 1 2\n"
        "inf inf 0 1\n"
        "inf inf inf 0"
    )
    hint_label = tk.Label(content_frame, text=hint_text, justify=tk.LEFT)
    hint_label.pack(anchor="w", pady=(8, 6))

    text_frame = tk.Frame(content_frame)
    text_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 12))

    scroll_bar = tk.Scrollbar(text_frame)
    scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)

    text_box = tk.Text(
        text_frame,
        width=78,
        height=10,
        font=("Consolas", 10),
        yscrollcommand=scroll_bar.set,
    )
    text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll_bar.config(command=text_box.yview)

    def on_submit():
        try:
            matrix = parse_matrix_input(size_entry.get(), text_box.get("1.0", tk.END))
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc), parent=root)
            return

        result["matrix"] = matrix
        result["mode"] = mode_var.get()
        root.destroy()

    button_frame = tk.Frame(main_frame)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))

    submit_button = tk.Button(
        button_frame,
        text="Confirm",
        width=12,
        command=on_submit,
    )
    submit_button.pack(anchor="center")

    root.mainloop()
    return result["matrix"], result["mode"]


def print_matrix(matrix):
    print("\nInput matrix:")
    for row in matrix:
        print(" ".join("inf" if value == math.inf else str(value) for value in row))


def show_planarity_result(lambda_edges, n):
    is_planar, underlying_graph = test_underlying_planarity(lambda_edges, n)

    draw_underlying_graph(underlying_graph, is_planar)

    if is_planar:
        print("The underlying graph is planar.")
    else:
        print("The underlying graph is not planar.")

    return is_planar, underlying_graph


def main():
    matrix, mode = get_user_configuration()

    if matrix is None:
        print("No matrix was entered. Program ended.")
        return

    print_matrix(matrix)
    exists, lambda_edges = foremost_realization_from_matrix_optimized(matrix)

    if not exists:
        print("Cannot generate a temporal graph.")
        return

    print("YES: Temporal graph realization exists")
    print("Temporal edges (v, w, t):")
    for (v, w), times in lambda_edges.items():
        for t in times:
            print(f"{v} -> {w} at time {t}")

    if mode == "planarity":
        show_planarity_result(lambda_edges, len(matrix))
    else:
        plot_temporal_graph(lambda_edges, len(matrix))


if __name__ == "__main__":
    main()

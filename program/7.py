import importlib.util
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath


def load_base_module():
    base_path = Path(__file__).with_name("6.py")
    spec = importlib.util.spec_from_file_location("temporal_graph_v6", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load 6.py as a base module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


EXAMPLE_MATRICES = {
    "no_temporal": {
        "size": 2,
        "description": (
            "Example 1: This matrix is too restrictive, so the algorithm cannot "
            "construct a valid foremost realization."
        ),
        "matrix": [
            [0, 0],
            [0, 0],
        ],
    },
    "temporal_not_planar": {
        "size": 5,
        "description": (
            "Example 2: The matrix can be realized as a temporal graph, but the "
            "underlying graph is not planar."
        ),
        "matrix": [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ],
    },
    "temporal_and_planar": {
        "size": 4,
        "description": (
            "Example 3: This matrix has a foremost realization and the resulting "
            "underlying graph is planar."
        ),
        "matrix": [
            [0, 1, 2, 3],
            [math.inf, 0, 1, 2],
            [math.inf, math.inf, 0, 1],
            [math.inf, math.inf, math.inf, 0],
        ],
    },
}


def test_planar_foremost_realization(matrix):
    """
    Check whether the matrix has a foremost realisation and whether that
    realisation is planar.
    """
    exists, lambda_edges = base.foremost_realization_from_matrix_optimized(matrix)
    if not exists:
        return False, False, None, None

    is_planar, underlying_graph = base.test_underlying_planarity(
        lambda_edges, len(matrix)
    )
    return True, is_planar, lambda_edges, underlying_graph


def draw_underlying_graph_no_show(underlying_graph, is_planar):
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
    return fig


def plot_temporal_graph_no_show(lambda_edges, n):
    curve_offset_ratio = 0.32
    node_size = 80
    node_radius_points = math.sqrt(node_size / math.pi)
    arrow_shrink = node_radius_points * 0.7
    label_offset_x = 0.03

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))

    for (u, v), times in lambda_edges.items():
        graph.add_edge(u, v, times=sorted(times))

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

            arrow_path = MplPath(
                [start, control, end],
                [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
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
    return fig


def get_user_configuration():
    result = {"matrix": None}

    root = tk.Tk()
    root.title("Temporal Graph Matrix Input")
    root.geometry("760x640")
    root.minsize(760, 640)

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)

    content_frame = tk.Frame(main_frame)
    content_frame.pack(fill=tk.BOTH, expand=True)

    title_label = tk.Label(
        content_frame,
        text="Enter the matrix and click Run Test",
        font=("Microsoft YaHei", 13, "bold"),
    )
    title_label.pack(pady=(0, 6), anchor="w")

    size_frame = tk.Frame(content_frame)
    size_frame.pack(pady=6, anchor="w")

    size_label = tk.Label(size_frame, text="Matrix size n:")
    size_label.pack(side=tk.LEFT, padx=(0, 8))

    size_entry = tk.Entry(size_frame, width=10)
    size_entry.pack(side=tk.LEFT)

    example_info_var = tk.StringVar(
        value="Click an example to auto-fill the matrix input area."
    )

    def fill_example(example_key):
        example = EXAMPLE_MATRICES[example_key]
        size_entry.delete(0, tk.END)
        size_entry.insert(0, str(example["size"]))
        text_box.delete("1.0", tk.END)
        for row in example["matrix"]:
            rendered_row = " ".join("inf" if value == math.inf else str(value) for value in row)
            text_box.insert(tk.END, rendered_row + "\n")
        example_info_var.set(example["description"])

    def run_test_from_ui():
        try:
            matrix = base.parse_matrix_input(size_entry.get(), text_box.get("1.0", tk.END))
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc), parent=root)
            return

        result["matrix"] = matrix
        root.destroy()

    example_frame = tk.LabelFrame(content_frame, text="Examples", padx=10, pady=8)
    example_frame.pack(fill=tk.X, pady=(4, 6))

    example_buttons = tk.Frame(example_frame)
    example_buttons.pack(fill=tk.X)

    tk.Button(
        example_buttons,
        text="Example 1: No temporal graph",
        command=lambda: fill_example("no_temporal"),
    ).pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

    tk.Button(
        example_buttons,
        text="Example 2: Temporal, not planar",
        command=lambda: fill_example("temporal_not_planar"),
    ).pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

    tk.Button(
        example_buttons,
        text="Example 3: Temporal and planar",
        command=lambda: fill_example("temporal_and_planar"),
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    example_info_label = tk.Label(
        content_frame,
        textvariable=example_info_var,
        justify=tk.LEFT,
        wraplength=700,
        fg="#333333",
    )
    example_info_label.pack(anchor="w", pady=(0, 8))

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

    button_frame = tk.Frame(main_frame)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2))

    submit_button = tk.Button(
        button_frame,
        text="Run Test",
        width=12,
        command=run_test_from_ui,
    )
    submit_button.pack(anchor="center")

    root.mainloop()
    return result["matrix"]


def print_matrix(matrix):
    print("\nInput matrix:")
    for row in matrix:
        print(" ".join("inf" if value == base.math.inf else str(value) for value in row))


def run_combined_test(matrix):
    exists, lambda_edges = base.foremost_realization_from_matrix_optimized(matrix)

    if not exists:
        print("NO: No foremost realization exists for this matrix.")
        return False, False, None, None

    print("YES: A temporal graph realization exists.")
    print("Temporal edges (v, w, t):")
    for (v, w), times in lambda_edges.items():
        for t in times:
            print(f"{v} -> {w} at time {t}")

    is_planar, underlying_graph = base.test_underlying_planarity(lambda_edges, len(matrix))
    if is_planar:
        print("YES: A planar temporal graph with a foremost realization exists.")
    else:
        print("NO: A foremost realization exists, but the resulting temporal graph is not planar.")

    return True, is_planar, lambda_edges, underlying_graph


def main():
    matrix = get_user_configuration()

    if matrix is None:
        print("No matrix was entered. Program ended.")
        return

    print_matrix(matrix)

    exists, is_planar, lambda_edges, underlying_graph = run_combined_test(matrix)
    if not exists:
        return

    draw_underlying_graph_no_show(underlying_graph, is_planar)
    plot_temporal_graph_no_show(lambda_edges, len(matrix))
    plt.show()


if __name__ == "__main__":
    main()

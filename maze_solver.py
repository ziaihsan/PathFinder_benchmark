"""
Maze Solver GUI with Tkinter

Features:
- Generates a random maze using recursive backtracker.
- Runs BFS, DFS, and A* sequentially on the same maze.
- Overlays the paths with different colors (BFS blue, DFS green, A* red).
- Shows performance metrics (steps and time in ms) in a table.
- Canvas automatically scales to show the whole maze within a window ≤ 800x800.

Functions exposed:
- generate_maze(rows, cols) -> (grid, start, end)
- bfs(grid, start, end) -> path
- dfs(grid, start, end) -> path
- astar(grid, start, end) -> path

Run:
    python -m py_compile maze_solver.py  # compile check
    python maze_solver.py                 # run the GUI
"""

from __future__ import annotations

import heapq
from array import array
import random
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as e:  # pragma: no cover - for environments without Tk
    raise


# Direction bit flags for cell walls
N, S, E, W = 1, 2, 4, 8
DIRS: Dict[int, Tuple[int, int]] = {
    N: (-1, 0),
    S: (1, 0),
    E: (0, 1),
    W: (0, -1),
}
OPPOSITE = {N: S, S: N, E: W, W: E}


def generate_maze(rows: int, cols: int) -> Tuple[List[List[int]], Tuple[int, int], Tuple[int, int]]:
    """Generate a maze using iterative recursive-backtracker (DFS) algorithm.

    Each cell stores bitflags of walls that are present. Initially all walls
    (N|S|E|W) are present; removing a wall clears the corresponding bit.

    Returns:
        grid: 2D list of ints (wall bitmasks)
        start: (row, col), default (0, 0)
        end: (row, col), default (rows-1, cols-1)
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    grid = [[N | S | E | W for _ in range(cols)] for _ in range(rows)]
    visited = [[False] * cols for _ in range(rows)]

    stack: List[Tuple[int, int]] = [(0, 0)]
    visited[0][0] = True

    while stack:
        r, c = stack[-1]
        # Find unvisited neighbors
        candidates: List[Tuple[int, int, int]] = []  # (dir_bit, nr, nc)
        for d, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                candidates.append((d, nr, nc))

        if candidates:
            d, nr, nc = random.choice(candidates)
            # Remove wall between current and neighbor
            grid[r][c] &= ~d
            grid[nr][nc] &= ~OPPOSITE[d]
            visited[nr][nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    # Optionally add extra loops for larger mazes to create multiple routes
    if rows >= 100 and cols >= 100:
        # Add a number of extra passages proportional to area but conservative
        area = rows * cols
        extra_ratio = 0.02  # 2% of cells open an extra wall
        extra = int(area * extra_ratio)
        _add_extra_passages(grid, extra)

    start = (0, 0)
    end = (rows - 1, cols - 1)
    return grid, start, end


def _add_extra_passages(grid: List[List[int]], extra: int) -> None:
    rows, cols = len(grid), len(grid[0])
    if extra <= 0:
        return
    for _ in range(extra):
        r = random.randrange(rows)
        c = random.randrange(cols)
        d = random.choice((N, S, E, W))
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            # If there is a wall between (r,c) and (nr,nc), remove it
            if grid[r][c] & d:
                grid[r][c] &= ~d
                grid[nr][nc] &= ~OPPOSITE[d]


def _neighbors(grid: List[List[int]], r: int, c: int) -> Iterable[Tuple[int, int]]:
    rows, cols = len(grid), len(grid[0])
    cell = grid[r][c]
    # If there is NO wall to that side, neighbor is reachable
    if c + 1 < cols and not (cell & E):
        yield r, c + 1
    if c - 1 >= 0 and not (cell & W):
        yield r, c - 1
    if r + 1 < rows and not (cell & S):
        yield r + 1, c
    if r - 1 >= 0 and not (cell & N):
        yield r - 1, c


def _reconstruct_path(came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
                      end: Tuple[int, int]) -> List[Tuple[int, int]]:
    if end not in came_from:
        return []
    path: List[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = end
    while cur is not None:
        path.append(cur)
        cur = came_from.get(cur)
    path.reverse()
    return path


def bfs(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Breadth-First Search path on the maze grid."""
    q: deque[Tuple[int, int]] = deque([start])
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    visited = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == end:
            break
        for nr, nc in _neighbors(grid, r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                came_from[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return _reconstruct_path(came_from, end)


def count_shortest_paths(
    grid: List[List[int]],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> int:
    """Count the number of distinct shortest paths from start to end.

    Uses BFS dynamic programming: for each node, accumulate number of ways to
    reach it along shortest paths. Works in O(V+E) time for unweighted graphs.
    """
    q: deque[Tuple[int, int]] = deque([start])
    dist: Dict[Tuple[int, int], int] = {start: 0}
    ways: Dict[Tuple[int, int], int] = {start: 1}

    end_dist: Optional[int] = None

    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        if end_dist is not None and d > end_dist:
            # All nodes at shortest distance processed
            break
        for nr, nc in _neighbors(grid, r, c):
            nd = d + 1
            if (nr, nc) not in dist:
                dist[(nr, nc)] = nd
                ways[(nr, nc)] = ways[(r, c)]
                q.append((nr, nc))
                if (nr, nc) == end and end_dist is None:
                    end_dist = nd
            elif nd == dist[(nr, nc)]:
                ways[(nr, nc)] = ways.get((nr, nc), 0) + ways[(r, c)]

    return ways.get(end, 0)


def dfs(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Depth-First Search path on the maze grid (iterative)."""
    stack: List[Tuple[int, int]] = [start]
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    visited = {start}
    while stack:
        r, c = stack.pop()
        if (r, c) == end:
            break
        for nr, nc in _neighbors(grid, r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                came_from[(nr, nc)] = (r, c)
                stack.append((nr, nc))
    return _reconstruct_path(came_from, end)


def astar(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """A* search with Manhattan heuristic."""
    def h(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap: List[Tuple[int, int, Tuple[int, int]]] = []
    counter = 0
    g: Dict[Tuple[int, int], int] = {start: 0}
    f_start = h(start, end)
    heapq.heappush(open_heap, (f_start, counter, start))
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    in_open = {start}

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == end:
            break

        in_open.discard(current)
        r, c = current
        for nr, nc in _neighbors(grid, r, c):
            tentative_g = g[current] + 1
            if tentative_g < g.get((nr, nc), 1_000_000_000):
                came_from[(nr, nc)] = current
                g[(nr, nc)] = tentative_g
                if (nr, nc) not in in_open:
                    counter += 1
                    heapq.heappush(open_heap, (tentative_g + h((nr, nc), end), counter, (nr, nc)))
                    in_open.add((nr, nc))

    return _reconstruct_path(came_from, end)


class MazeApp:
    """Tkinter GUI application for maze generation and solving."""

    MAX_WINDOW = 800
    PANEL_WIDTH = 260
    CANVAS_MARGIN = 8

    COLORS = {
        "BFS": "#1E90FF",   # DodgerBlue
        "DFS": "#2E8B57",   # SeaGreen
        "A*": "#FF4500",    # OrangeRed
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Maze Solver (BFS / DFS / A*)")

        # Fix overall window size to ≤ 800x800
        self.root.geometry(f"{self.MAX_WINDOW}x{self.MAX_WINDOW}")
        self.root.minsize(self.MAX_WINDOW, self.MAX_WINDOW)
        self.root.maxsize(self.MAX_WINDOW, self.MAX_WINDOW)

        # Left canvas and right panel
        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        # Canvas area: allow it to expand to use available space
        self.canvas_size = min(self.MAX_WINDOW - self.PANEL_WIDTH - 10, self.MAX_WINDOW - 10)
        self.canvas = tk.Canvas(container, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        panel = tk.Frame(container, width=self.PANEL_WIDTH)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        panel.pack_propagate(False)

        # Controls
        lbl_title = tk.Label(panel, text="Pengaturan", font=("Segoe UI", 12, "bold"))
        lbl_title.pack(anchor="w", pady=(0, 8))

        size_frame = tk.Frame(panel)
        size_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(size_frame, text="Ukuran Labirin (N × N)").pack(anchor="w")
        self.size_var = tk.IntVar(value=20)
        self.size_scale = tk.Scale(
            size_frame,
            from_=2,
            to=1000,
            resolution=2,  # even numbers only
            orient=tk.HORIZONTAL,
            variable=self.size_var,
            length=self.PANEL_WIDTH - 20,
        )
        self.size_scale.pack(anchor="w")

        self.solve_btn = tk.Button(panel, text="Solve", command=self.on_solve)
        self.solve_btn.pack(anchor="w", pady=(0, 10), fill=tk.X)

        # Results table
        tk.Label(panel, text="Hasil (setiap klik Solve)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        columns = ("alg", "paths", "steps", "time")
        self.table = ttk.Treeview(panel, columns=columns, show="headings", height=6)
        self.table.heading("alg", text="Algoritma")
        self.table.heading("paths", text="PathFound")
        self.table.heading("steps", text="Langkah")
        self.table.heading("time", text="Waktu (ms)")
        # Make columns compact so all are visible by default
        self.table.column("alg", width=60, minwidth=50, anchor=tk.W, stretch=False)
        self.table.column("paths", width=60, minwidth=50, anchor=tk.CENTER, stretch=False)
        self.table.column("steps", width=60, minwidth=50, anchor=tk.CENTER, stretch=False)
        self.table.column("time", width=60, minwidth=50, anchor=tk.E, stretch=False)
        self.table.pack(fill=tk.X, pady=(5, 0))

        # Storage for current maze and drawing params
        self.grid: Optional[List[List[int]]] = None
        self.start: Optional[Tuple[int, int]] = None
        self.end: Optional[Tuple[int, int]] = None
        self.cell_px: float = 0.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.paths: Dict[str, List[Tuple[int, int]]] = {}
        self.fast_draw: bool = False

        # Redraw maze responsively when canvas size changes
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.root.update_idletasks()

    def on_solve(self) -> None:
        # Generate maze for current slider size (ensure even)
        n = int(self.size_var.get())
        if n % 2 != 0:
            n -= 1
            n = max(2, n)
            self.size_var.set(n)

        self.grid, self.start, self.end = generate_maze(n, n)
        self._prepare_draw_params(n, n)
        self._draw_maze()

        # Run solvers and time them
        assert self.grid is not None and self.start is not None and self.end is not None
        results: List[Tuple[str, List[Tuple[int, int]], float]] = []  # (name, path, ms)

        t0 = time.perf_counter()
        path_bfs = bfs(self.grid, self.start, self.end)
        t1 = time.perf_counter()
        results.append(("BFS", path_bfs, (t1 - t0) * 1000.0))

        t0 = time.perf_counter()
        path_dfs = dfs(self.grid, self.start, self.end)
        t1 = time.perf_counter()
        results.append(("DFS", path_dfs, (t1 - t0) * 1000.0))

        t0 = time.perf_counter()
        path_astar = astar(self.grid, self.start, self.end)
        t1 = time.perf_counter()
        results.append(("A*", path_astar, (t1 - t0) * 1000.0))

        # Count number of distinct shortest paths (for overview visibility)
        try:
            num_paths = count_shortest_paths(self.grid, self.start, self.end)
        except Exception:
            num_paths = 0

        # Draw paths (overlay) in order BFS, DFS, A*
        self.paths = {"BFS": path_bfs, "DFS": path_dfs, "A*": path_astar}
        self._draw_path(path_bfs, self.COLORS["BFS"])  # blue
        self._draw_path(path_dfs, self.COLORS["DFS"])  # green
        self._draw_path(path_astar, self.COLORS["A*"])  # red

        # Update table
        self._update_table(results, num_paths)

    def _prepare_draw_params(self, rows: int, cols: int) -> None:
        # Compute floating cell size to make maze fill the white canvas area
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        if w <= 1 or h <= 1:
            w = h = self.canvas_size
        m = self.CANVAS_MARGIN
        usable_w = max(1, w - 2 * m)
        usable_h = max(1, h - 2 * m)
        size = min(usable_w / max(1, cols), usable_h / max(1, rows))
        maze_w = cols * size
        maze_h = rows * size
        # Center the maze inside the canvas
        self.offset_x = (w - maze_w) / 2
        self.offset_y = (h - maze_h) / 2
        self.cell_px = size

    def _clear_canvas(self) -> None:
        self.canvas.delete("all")

    def _draw_maze(self) -> None:
        if self.grid is None:
            return
        self._clear_canvas()
        rows, cols = len(self.grid), len(self.grid[0])
        self._prepare_draw_params(rows, cols)
        size = self.cell_px
        # Decide fast draw mode for large or tiny-cell mazes
        self.fast_draw = (rows > 200 or cols > 200 or size < 2.0)

        # Outer border aligned to computed offsets
        x0, y0 = self.offset_x, self.offset_y
        x1, y1 = x0 + cols * size, y0 + rows * size
        self.canvas.create_rectangle(x0, y0, x1, y1)

        # Draw internal walls unless in fast mode (too many items)
        if not self.fast_draw:
            for r in range(rows):
                for c in range(cols):
                    cell = self.grid[r][c]
                    cx0, cy0 = x0 + c * size, y0 + r * size
                    cx1, cy1 = cx0 + size, cy0 + size
                    if cell & E:
                        self.canvas.create_line(cx1, cy0, cx1, cy1)
                    if cell & S:
                        self.canvas.create_line(cx0, cy1, cx1, cy1)

        # Mark start and end
        if self.start and self.end:
            sx = x0 + self.start[1] * size + size / 2
            sy = y0 + self.start[0] * size + size / 2
            ex = x0 + self.end[1] * size + size / 2
            ey = y0 + self.end[0] * size + size / 2
            r = max(2.0, size / 4)
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill="#FFD700", outline="")  # gold
            self.canvas.create_oval(ex - r, ey - r, ex + r, ey + r, fill="#DC143C", outline="")  # crimson

    def _draw_path(self, path: List[Tuple[int, int]], color: str) -> None:
        if not path:
            return
        size = self.cell_px
        x0, y0 = self.offset_x, self.offset_y
        w = max(1, int(round(size / 3)))
        # Downsample very long polylines to keep UI responsive
        max_points = 6000 if not self.fast_draw else 2000
        step = max(1, len(path) // max_points)
        points: List[float] = []
        for idx, (r, c) in enumerate(path):
            if idx % step != 0 and idx != len(path) - 1:
                continue
            x = x0 + c * size + size / 2
            y = y0 + r * size + size / 2
            points.extend([x, y])
        # Draw as a single polyline for speed
        self.canvas.create_line(*points, fill=color, width=w, capstyle=tk.ROUND, joinstyle=tk.ROUND)

    def _update_table(self, results: List[Tuple[str, List[Tuple[int, int]], float]], num_paths: int) -> None:
        # Clear existing
        for item in self.table.get_children():
            self.table.delete(item)
        # Insert new rows
        for name, path, ms in results:
            steps = max(0, len(path) - 1) if path else 0
            self.table.insert("", tk.END, values=(name, num_paths, steps, f"{ms:.3f}"))

    def _on_canvas_resize(self, event: tk.Event) -> None:  # type: ignore[name-defined]
        # Redraw maze and any existing paths to fit the new size
        if self.grid is None:
            return
        self._draw_maze()
        # Redraw cached paths if present
        for name, path in self.paths.items():
            color = self.COLORS.get(name, "black")
            self._draw_path(path, color)


# Optimized solver implementations appended below to improve performance,
# especially for mazes larger than 200x200.

def _reconstruct_path_idx(prev: array, end_idx: int, cols: int) -> List[Tuple[int, int]]:  # type: ignore[no-redef]
    path_idx: List[int] = []
    cur = end_idx
    while cur != -1:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx.reverse()
    return [(i // cols, i % cols) for i in path_idx]


def bfs(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:  # type: ignore[no-redef]
    rows, cols = len(grid), len(grid[0])
    total = rows * cols
    def idx(r: int, c: int) -> int:
        return r * cols + c
    start_i = idx(*start)
    end_i = idx(*end)

    q: deque[int] = deque([start_i])
    visited = bytearray(total)
    visited[start_i] = 1
    prev = array('i', [-1] * total)

    while q:
        i = q.popleft()
        if i == end_i:
            break
        r, c = divmod(i, cols)
        cell = grid[r][c]
        if c + 1 < cols and not (cell & E):
            ni = i + 1
            if not visited[ni]:
                visited[ni] = 1
                prev[ni] = i
                q.append(ni)
        if c - 1 >= 0 and not (cell & W):
            ni = i - 1
            if not visited[ni]:
                visited[ni] = 1
                prev[ni] = i
                q.append(ni)
        if r + 1 < rows and not (cell & S):
            ni = i + cols
            if not visited[ni]:
                visited[ni] = 1
                prev[ni] = i
                q.append(ni)
        if r - 1 >= 0 and not (cell & N):
            ni = i - cols
            if not visited[ni]:
                visited[ni] = 1
                prev[ni] = i
                q.append(ni)

    if end_i != start_i and prev[end_i] == -1:
        return []
    return _reconstruct_path_idx(prev, end_i, cols)


def dfs(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:  # type: ignore[no-redef]
    rows, cols = len(grid), len(grid[0])
    total = rows * cols
    def idx(r: int, c: int) -> int:
        return r * cols + c
    start_i = idx(*start)
    end_i = idx(*end)

    stack: List[int] = [start_i]
    visited = bytearray(total)
    visited[start_i] = 1
    prev = array('i', [-1] * total)

    while stack:
        i = stack.pop()
        if i == end_i:
            break
        r, c = divmod(i, cols)
        cell = grid[r][c]
        if c + 1 < cols and not (cell & E):
            ni = i + 1
            if not visited[ni]:
                visited[ni] = 1
                prev[ni] = i
                stack.append(ni)
        if c - 1 >= 0 and not (cell & W):
            ni = i - 1
            if not visited[ni]:
                visited[ni] = 1
                prev[ni] = i
                stack.append(ni)
        if r + 1 < rows and not (cell & S):
            ni = i + cols
            if not visited[ni]:
                visited[ni] = 1
                prev[ni] = i
                stack.append(ni)
        if r - 1 >= 0 and not (cell & N):
            ni = i - cols
            if not visited[ni]:
                visited[ni] = 1
                prev[ni] = i
                stack.append(ni)

    if end_i != start_i and prev[end_i] == -1:
        return []
    return _reconstruct_path_idx(prev, end_i, cols)


def astar(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:  # type: ignore[no-redef]
    rows, cols = len(grid), len(grid[0])
    total = rows * cols
    def idx(r: int, c: int) -> int:
        return r * cols + c
    start_i = idx(*start)
    end_i = idx(*end)

    def h_idx(a: int, b: int) -> int:
        ar, ac = divmod(a, cols)
        br, bc = divmod(b, cols)
        return abs(ar - br) + abs(ac - bc)

    g = array('i', [10**9] * total)
    g[start_i] = 0
    prev = array('i', [-1] * total)
    open_heap: List[Tuple[int, int, int]] = []
    counter = 0
    heapq.heappush(open_heap, (h_idx(start_i, end_i), counter, start_i))
    in_open = bytearray(total)
    in_open[start_i] = 1

    while open_heap:
        _, _, i = heapq.heappop(open_heap)
        if i == end_i:
            break
        in_open[i] = 0
        r, c = divmod(i, cols)
        cell = grid[r][c]

        if c + 1 < cols and not (cell & E):
            ni = i + 1
            tentative = g[i] + 1
            if tentative < g[ni]:
                g[ni] = tentative
                prev[ni] = i
                if not in_open[ni]:
                    counter += 1
                    heapq.heappush(open_heap, (tentative + h_idx(ni, end_i), counter, ni))
                    in_open[ni] = 1
        if c - 1 >= 0 and not (cell & W):
            ni = i - 1
            tentative = g[i] + 1
            if tentative < g[ni]:
                g[ni] = tentative
                prev[ni] = i
                if not in_open[ni]:
                    counter += 1
                    heapq.heappush(open_heap, (tentative + h_idx(ni, end_i), counter, ni))
                    in_open[ni] = 1
        if r + 1 < rows and not (cell & S):
            ni = i + cols
            tentative = g[i] + 1
            if tentative < g[ni]:
                g[ni] = tentative
                prev[ni] = i
                if not in_open[ni]:
                    counter += 1
                    heapq.heappush(open_heap, (tentative + h_idx(ni, end_i), counter, ni))
                    in_open[ni] = 1
        if r - 1 >= 0 and not (cell & N):
            ni = i - cols
            tentative = g[i] + 1
            if tentative < g[ni]:
                g[ni] = tentative
                prev[ni] = i
                if not in_open[ni]:
                    counter += 1
                    heapq.heappush(open_heap, (tentative + h_idx(ni, end_i), counter, ni))
                    in_open[ni] = 1

    if end_i != start_i and prev[end_i] == -1:
        return []
    return _reconstruct_path_idx(prev, end_i, cols)


def count_shortest_paths(
    grid: List[List[int]],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> int:  # type: ignore[no-redef]
    rows, cols = len(grid), len(grid[0])
    total = rows * cols
    def idx(r: int, c: int) -> int:
        return r * cols + c
    start_i = idx(*start)
    end_i = idx(*end)

    INF = 10**9
    dist = array('i', [INF] * total)
    ways = array('Q', [0] * total)
    q: deque[int] = deque([start_i])
    dist[start_i] = 0
    ways[start_i] = 1
    end_dist: Optional[int] = None

    while q:
        i = q.popleft()
        d = dist[i]
        if end_dist is not None and d > end_dist:
            break
        r, c = divmod(i, cols)
        cell = grid[r][c]
        if c + 1 < cols and not (cell & E):
            ni = i + 1
            nd = d + 1
            if dist[ni] == INF:
                dist[ni] = nd
                ways[ni] = ways[i]
                q.append(ni)
                if ni == end_i and end_dist is None:
                    end_dist = nd
            elif nd == dist[ni]:
                ways[ni] += ways[i]
        if c - 1 >= 0 and not (cell & W):
            ni = i - 1
            nd = d + 1
            if dist[ni] == INF:
                dist[ni] = nd
                ways[ni] = ways[i]
                q.append(ni)
                if ni == end_i and end_dist is None:
                    end_dist = nd
            elif nd == dist[ni]:
                ways[ni] += ways[i]
        if r + 1 < rows and not (cell & S):
            ni = i + cols
            nd = d + 1
            if dist[ni] == INF:
                dist[ni] = nd
                ways[ni] = ways[i]
                q.append(ni)
                if ni == end_i and end_dist is None:
                    end_dist = nd
            elif nd == dist[ni]:
                ways[ni] += ways[i]
        if r - 1 >= 0 and not (cell & N):
            ni = i - cols
            nd = d + 1
            if dist[ni] == INF:
                dist[ni] = nd
                ways[ni] = ways[i]
                q.append(ni)
                if ni == end_i and end_dist is None:
                    end_dist = nd
            elif nd == dist[ni]:
                ways[ni] += ways[i]

    return int(ways[end_i])


def main() -> None:
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

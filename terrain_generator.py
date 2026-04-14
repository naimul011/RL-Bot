"""
3D terrain generation for PyBullet robot simulation.

Creates heightfield terrains of various types, loads them into a PyBullet
physics client, and exposes height/slope queries used by the RL environment
to compute slope-dependent motor lag.

Terrain types:
  flat   – flat plane (paper baseline; zero slope everywhere)
  hills  – multiple overlapping Gaussian bumps (moderate complexity)
  ramps  – linear ramp in the x-direction (~2.5% grade)
  mixed  – hills + gentle diagonal slope (hardest; used for 3D RL training)

Usage:
    gen = TerrainGenerator(size=20.0, resolution=128, seed=42)
    body_id, heights = gen.create(client_id, terrain_type='hills')
    h  = gen.get_height_at(x, y)
    sx, sy = gen.get_slope_at(x, y)
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import pybullet as p
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3d projection)


class TerrainGenerator:
    """
    Manages a discrete heightfield terrain for PyBullet simulation.

    The terrain is a square of `size × size` metres, sampled on a
    `resolution × resolution` grid.  After calling :meth:`create` the
    heightfield lives inside PyBullet; queries to :meth:`get_height_at`
    and :meth:`get_slope_at` are answered by bilinear/finite-difference
    interpolation of the stored numpy array — no additional PyBullet
    raycasts are needed.

    Coordinate conventions
    ----------------------
    World origin (0, 0) maps to the *centre* of the heightfield.
    Grid indices (row, col) satisfy:

        x_world = (col - (N-1)/2) * scale
        y_world = (row - (N-1)/2) * scale

    so  heights[row, col]  is the terrain elevation at world (x, y).
    """

    def __init__(
        self,
        size: float = 20.0,        # terrain width/height in metres
        resolution: int = 128,     # grid points per side (power-of-2 preferred)
        seed: int = 42,
    ):
        self.size = size
        self.resolution = resolution
        self.scale = size / (resolution - 1)   # metres per grid cell
        self.rng = np.random.RandomState(seed)

        self.heights: np.ndarray = None   # (N, N) float32, set by create()
        self._body_id: int = None
        self._client_id: int = None
        # PyBullet centres the heightfield bounding box at the body position,
        # i.e. world_z = base_z + (h - (h_min+h_max)/2).  We set
        # base_z = (h_min+h_max)/2  so that world_z == raw_h everywhere,
        # which keeps get_height_at() simple and correct.
        self._z_base: float = 0.0   # body z passed to createMultiBody (set in create())

    # ─────────────────────────────── Public API ───────────────────────────────

    def create(self, client_id: int, terrain_type: str = 'hills') -> tuple:
        """
        Build the terrain, load it into PyBullet, and return its body ID.

        Parameters
        ----------
        client_id    : PyBullet physics client ID
        terrain_type : 'flat' | 'hills' | 'ramps' | 'mixed'

        Returns
        -------
        (body_id, heights_2d)  where heights_2d has shape (N, N).
        """
        self._client_id = client_id
        N = self.resolution

        self.heights = self._generate_heights(terrain_type)

        # PyBullet heightfield: data is a flat list in row-major order
        # i.e. data[row * N + col] = heights[row, col]
        hf_data = self.heights.flatten().tolist()

        shape_id = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD,
            meshScale=[self.scale, self.scale, 1.0],
            heightfieldData=hf_data,
            numHeightfieldRows=N,
            numHeightfieldColumns=N,
            physicsClientId=client_id,
        )

        # PyBullet internally centres the heightfield so that
        #   world_z = base_z + (h - (h_min + h_max)/2)
        # Setting base_z = (h_min + h_max)/2 makes world_z == raw_h,
        # which keeps get_height_at() trivial and correct.
        h_min = float(self.heights.min())
        h_max = float(self.heights.max())
        self._z_base = (h_min + h_max) * 0.5

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape_id,
            basePosition=[0.0, 0.0, self._z_base],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=client_id,
        )

        # Terrain colour
        colour_map = {
            'flat':  [0.50, 0.80, 0.35, 1.0],   # light green
            'hills': [0.35, 0.68, 0.22, 1.0],   # medium green
            'ramps': [0.62, 0.52, 0.30, 1.0],   # sandy brown
            'mixed': [0.30, 0.60, 0.20, 1.0],   # forest green
        }
        rgba = colour_map.get(terrain_type, [0.5, 0.7, 0.3, 1.0])
        p.changeVisualShape(body_id, -1, rgbaColor=rgba, physicsClientId=client_id)

        # High friction on the terrain surface
        p.changeDynamics(
            body_id, -1,
            lateralFriction=0.9,
            spinningFriction=0.05,
            rollingFriction=0.01,
            physicsClientId=client_id,
        )

        self._body_id = body_id
        return body_id, self.heights

    def get_height_at(self, x: float, y: float) -> float:
        """
        Bilinearly interpolate the terrain elevation at world position (x, y).

        Returns 0.0 if the terrain has not been generated yet.
        """
        if self.heights is None:
            return 0.0

        N = self.resolution
        s = self.scale

        # Fractional grid indices (col ≡ x-axis, row ≡ y-axis)
        col_f = x / s + (N - 1) * 0.5
        row_f = y / s + (N - 1) * 0.5

        col_f = float(np.clip(col_f, 0.0, N - 1 - 1e-9))
        row_f = float(np.clip(row_f, 0.0, N - 1 - 1e-9))

        c0, r0 = int(col_f), int(row_f)
        c1, r1 = c0 + 1, r0 + 1
        dc, dr = col_f - c0, row_f - r0

        h = (self.heights[r0, c0] * (1.0 - dc) * (1.0 - dr)
             + self.heights[r0, c1] * dc         * (1.0 - dr)
             + self.heights[r1, c0] * (1.0 - dc) * dr
             + self.heights[r1, c1] * dc         * dr)
        return float(h)

    def get_slope_at(self, x: float, y: float) -> tuple:
        """
        Central-difference estimate of terrain gradient at (x, y).

        Returns
        -------
        (slope_x, slope_y) : dh/dx, dh/dy  (dimensionless rise/run)
        """
        d = self.scale
        sx = (self.get_height_at(x + d, y) - self.get_height_at(x - d, y)) / (2.0 * d)
        sy = (self.get_height_at(x, y + d) - self.get_height_at(x, y - d)) / (2.0 * d)
        return float(sx), float(sy)

    def max_height(self) -> float:
        """Return the maximum terrain elevation."""
        if self.heights is None:
            return 0.0
        return float(self.heights.max())

    # ──────────────────────────── Visualisation ───────────────────────────────

    def visualize_2d(self, save_path: str = None, title: str = 'Terrain (top-down)') -> None:
        """Render the heightfield as a 2-D colour map."""
        if self.heights is None:
            print("[TerrainGenerator] No terrain to visualize – call create() first.")
            return

        half = self.size / 2.0
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            self.heights,
            origin='lower',
            extent=[-half, half, -half, half],
            cmap='terrain',
            interpolation='bilinear',
        )
        plt.colorbar(im, ax=ax, label='Height (m)')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(title)

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def visualize_3d(
        self,
        robot_paths: dict = None,   # {label: {'x': arr, 'y': arr}}
        save_path: str = None,
        title: str = '3D Terrain Navigation',
        downsample: int = 4,        # plot every Nth grid point for speed
    ) -> None:
        """
        Render the terrain as a 3-D surface with optional robot path overlays.

        Parameters
        ----------
        robot_paths : dict mapping label → {'x': array, 'y': array}
        save_path   : file path to save (PNG).  None → show interactively.
        downsample  : stride for surface mesh (reduces plot file size).
        """
        if self.heights is None:
            print("[TerrainGenerator] No terrain to visualize – call create() first.")
            return

        N = self.resolution
        d = downsample
        half = self.size / 2.0

        x1d = np.linspace(-half, half, N)
        y1d = np.linspace(-half, half, N)
        X, Y = np.meshgrid(x1d[::d], y1d[::d])
        Z = self.heights[::d, ::d]

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(
            X, Y, Z,
            cmap='terrain',
            alpha=0.75,
            linewidth=0,
            antialiased=True,
        )

        colours = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
        if robot_paths:
            for i, (label, path) in enumerate(robot_paths.items()):
                px = np.asarray(path['x'])
                py = np.asarray(path['y'])
                pz = np.array([self.get_height_at(xi, yi) + 0.13
                               for xi, yi in zip(px, py)])
                c = colours[i % len(colours)]
                ax.plot(px, py, pz, color=c, linewidth=2.5, label=label, zorder=5)

            ax.legend(loc='upper left', fontsize=10)

        ax.set_xlabel('x (m)', labelpad=8)
        ax.set_ylabel('y (m)', labelpad=8)
        ax.set_zlabel('z (m)', labelpad=8)
        ax.set_title(title, fontsize=12)
        ax.view_init(elev=28, azim=-55)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    # ─────────────────────────── Internal helpers ─────────────────────────────

    def _generate_heights(self, terrain_type: str) -> np.ndarray:
        N = self.resolution
        rng = self.rng
        H = np.zeros((N, N), dtype=np.float32)

        if terrain_type == 'flat':
            pass   # all zeros

        elif terrain_type == 'hills':
            # 8 overlapping Gaussian bumps, smoothed to avoid discontinuities
            ii = np.arange(N, dtype=np.float32)
            jj = np.arange(N, dtype=np.float32)
            II, JJ = np.meshgrid(ii, jj, indexing='ij')   # II=row, JJ=col
            for _ in range(8):
                cr = rng.uniform(0.2 * N, 0.8 * N)
                cc = rng.uniform(0.2 * N, 0.8 * N)
                amp = rng.uniform(0.15, 0.45)
                sr = rng.uniform(8, 20)
                sc = rng.uniform(8, 20)
                H += amp * np.exp(
                    -((II - cr)**2 / (2 * sr**2) + (JJ - cc)**2 / (2 * sc**2))
                )
            H = gaussian_filter(H, sigma=2.5)

        elif terrain_type == 'ramps':
            # Gentle ramp in the x-direction: 0.50 m rise over 20 m (~2.5% grade)
            col_norm = np.linspace(0.0, 1.0, N, dtype=np.float32)
            H = np.tile(col_norm, (N, 1)) * 0.50  # H[row, col]

        elif terrain_type == 'mixed':
            ii = np.arange(N, dtype=np.float32)
            jj = np.arange(N, dtype=np.float32)
            II, JJ = np.meshgrid(ii, jj, indexing='ij')
            # 5 Gaussian hills
            for _ in range(5):
                cr = rng.uniform(0.15 * N, 0.85 * N)
                cc = rng.uniform(0.15 * N, 0.85 * N)
                amp = rng.uniform(0.10, 0.35)
                sr = rng.uniform(10, 22)
                sc = rng.uniform(10, 22)
                H += amp * np.exp(
                    -((II - cr)**2 / (2 * sr**2) + (JJ - cc)**2 / (2 * sc**2))
                )
            # Gentle diagonal slope
            col_norm = np.linspace(0.0, 1.0, N, dtype=np.float32)
            row_norm = np.linspace(0.0, 1.0, N, dtype=np.float32)
            slope_grid = np.outer(row_norm, np.ones(N)) * 0.15 + np.outer(np.ones(N), col_norm) * 0.10
            H += slope_grid.astype(np.float32)
            H = gaussian_filter(H, sigma=1.8)

        else:
            raise ValueError(
                f"Unknown terrain_type='{terrain_type}'. "
                "Choose from: 'flat', 'hills', 'ramps', 'mixed'."
            )

        return H


# ──────────────────────────── Quick standalone test ───────────────────────────
if __name__ == '__main__':
    import os
    os.makedirs('./results', exist_ok=True)

    for ttype in ['flat', 'hills', 'ramps', 'mixed']:
        gen = TerrainGenerator(size=20.0, resolution=128, seed=42)
        gen.heights = gen._generate_heights(ttype)
        gen.visualize_2d(save_path=f'./results/terrain_2d_{ttype}.png', title=f'Terrain: {ttype}')
        gen.visualize_3d(save_path=f'./results/terrain_3d_{ttype}.png', title=f'3D Terrain: {ttype}')
        print(f"[{ttype}]  max_height={gen.heights.max():.3f}m  saved to ./results/")

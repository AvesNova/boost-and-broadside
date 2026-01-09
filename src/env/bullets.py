import numpy as np


class Bullets:
    """
    Manages a pool of bullets using pre-allocated numpy arrays for performance.

    This class handles the lifecycle of bullets, including creation, movement,
    and removal. It uses a swap-and-pop strategy to keep active bullets
    contiguous in memory, avoiding expensive array resizing.
    """

    def __init__(self, max_bullets: int) -> None:
        """
        Initialize the bullet manager.

        Args:
            max_bullets: Maximum number of bullets allowed simultaneously.
        """
        self.num_active = 0
        self.max_bullets = max_bullets

        # Pre-allocate arrays for bullet properties
        self.x = np.zeros(self.max_bullets, dtype=np.float32)
        self.y = np.zeros(self.max_bullets, dtype=np.float32)
        self.vx = np.zeros(self.max_bullets, dtype=np.float32)
        self.vy = np.zeros(self.max_bullets, dtype=np.float32)
        self.time_remaining = np.zeros(self.max_bullets, dtype=np.float32)
        self.ship_id = np.zeros(self.max_bullets, dtype=np.uint8)

        # Free list for O(1) allocation from inactive region
        # Stores indices of slots that are currently not in use
        self.free_slots = np.arange(self.max_bullets - 1, -1, -1, dtype=np.int32)
        self.num_free = self.max_bullets

    def add_bullet(
        self, ship_id: int, x: float, y: float, vx: float, vy: float, lifetime: float
    ) -> int:
        """
        Add a new bullet to the system.

        Args:
            ship_id: ID of the ship that fired the bullet.
            x: Initial X position.
            y: Initial Y position.
            vx: Initial X velocity.
            vy: Initial Y velocity.
            lifetime: Time in seconds before the bullet expires.

        Returns:
            The index of the newly added bullet, or -1 if no slots are free.
        """
        if self.num_free == 0:
            return -1

        # Get a free slot from the free list
        self.num_free -= 1
        slot = self.free_slots[self.num_free]

        # Ensure the slot is within the active region [0, num_active]
        # If the slot is outside, swap it with the element at the end of the active region.
        # This maintains the invariant that active bullets are contiguous at the start.
        if slot >= self.num_active:
            active_end = self.num_active
            if slot != active_end:
                self._swap_bullets(slot, active_end)
                slot = active_end

        # Initialize bullet data
        self.x[slot] = x
        self.y[slot] = y
        self.vx[slot] = vx
        self.vy[slot] = vy
        self.time_remaining[slot] = lifetime
        self.ship_id[slot] = ship_id

        self.num_active += 1
        return slot

    def _swap_bullets(self, i: int, j: int) -> None:
        """
        Swap two bullets in all internal arrays.

        Args:
            i: Index of the first bullet.
            j: Index of the second bullet.
        """
        self.x[i], self.x[j] = self.x[j], self.x[i]
        self.y[i], self.y[j] = self.y[j], self.y[i]
        self.vx[i], self.vx[j] = self.vx[j], self.vx[i]
        self.vy[i], self.vy[j] = self.vy[j], self.vy[i]
        self.time_remaining[i], self.time_remaining[j] = (
            self.time_remaining[j],
            self.time_remaining[i],
        )
        self.ship_id[i], self.ship_id[j] = self.ship_id[j], self.ship_id[i]

    def remove_bullet(self, idx: int) -> None:
        """
        Remove a bullet by index.

        Args:
            idx: Index of the bullet to remove.
        """
        if idx >= self.num_active:
            return  # Already inactive

        self.num_active -= 1

        # Swap with last active bullet to keep active region contiguous
        if idx != self.num_active:
            self._swap_bullets(idx, self.num_active)

        # Return the slot to the free list
        self.free_slots[self.num_free] = self.num_active
        self.num_free += 1

    def update_all(self, dt: float) -> None:
        """
        Update positions and lifetimes of all active bullets.

        Args:
            dt: Time step in seconds.
        """
        if self.num_active == 0:
            return

        # Vectorized update for all active bullets
        active_slice = slice(0, self.num_active)
        self.x[active_slice] += self.vx[active_slice] * dt
        self.y[active_slice] += self.vy[active_slice] * dt
        self.time_remaining[active_slice] -= dt

        # Remove expired bullets
        self._remove_expired()

    def _remove_expired(self) -> None:
        """
        Identify and remove bullets that have exceeded their lifetime.
        """
        if self.num_active == 0:
            return

        # Find expired bullets
        expired_mask = self.time_remaining[: self.num_active] <= 0

        if not np.any(expired_mask):
            return

        # Get indices of bullets to keep
        keep_mask = ~expired_mask
        keep_indices = np.where(keep_mask)[0]
        new_active_count = len(keep_indices)

        if new_active_count == 0:
            # All bullets expired
            self.free_slots[: self.num_active] = np.arange(self.num_active)
            self.num_free = self.max_bullets
            self.num_active = 0
            return

        # Compact arrays - move kept bullets to front
        self.x[:new_active_count] = self.x[keep_indices]
        self.y[:new_active_count] = self.y[keep_indices]
        self.vx[:new_active_count] = self.vx[keep_indices]
        self.vy[:new_active_count] = self.vy[keep_indices]
        self.time_remaining[:new_active_count] = self.time_remaining[keep_indices]
        self.ship_id[:new_active_count] = self.ship_id[keep_indices]

        # Update free list with the newly freed slots
        expired_count = self.num_active - new_active_count
        self.free_slots[self.num_free : self.num_free + expired_count] = np.arange(
            new_active_count, self.num_active
        )
        self.num_free += expired_count
        self.num_active = new_active_count

    def get_active_positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the current state of all active bullets.

        Returns:
            Tuple containing:
            - x positions (np.ndarray)
            - y positions (np.ndarray)
            - ship IDs (np.ndarray)
        """
        if self.num_active == 0:
            return np.array([]), np.array([]), np.array([])

        return (
            self.x[: self.num_active],
            self.y[: self.num_active],
            self.ship_id[: self.num_active],
        )

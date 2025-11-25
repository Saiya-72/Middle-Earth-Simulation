import os
import time
import random
from typing import Any, Optional, List
import numpy as np

ROWS = 7   # map height
COLS = 12  # map width

# -----------------------------
# Utility
# -----------------------------

def cls() -> None:
    os.system("cls" if os.name == "nt" else "clear")


# -----------------------------
# Data Structures
# -----------------------------

class LinkedListNode:
    def __init__(self, value: Any, next_node: Optional["LinkedListNode"] = None):
        self.value = value
        self.next = next_node


class LinkedList:
    """Singly linked list used for the global collection of entities."""
    def __init__(self) -> None:
        self.head: Optional[LinkedListNode] = None

    def append(self, value: Any) -> None:
        node = LinkedListNode(value)
        if self.head is None:
            self.head = node
            return
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = node

    def remove_if(self, predicate) -> None:
        dummy = LinkedListNode(None, self.head)
        prev = dummy
        cur = self.head
        while cur is not None:
            if predicate(cur.value):
                prev.next = cur.next
            else:
                prev = cur
            cur = cur.next
        self.head = dummy.next

    def __iter__(self):
        cur = self.head
        while cur is not None:
            yield cur.value
            cur = cur.next

    def is_empty(self) -> bool:
        return self.head is None


class Stack:
    """Simple LIFO stack used for logging events each year."""
    def __init__(self) -> None:
        self._data: List[Any] = []

    def push(self, item: Any) -> None:
        self._data.append(item)

    def pop(self) -> Any:
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def is_empty(self) -> bool:
        return len(self._data) == 0


class Queue:
    """Circular-buffer queue used when scheduling entities into the priority queue."""
    def __init__(self) -> None:
        self._data: List[Optional[Any]] = [None] * 8
        self._head = 0
        self._tail = 0
        self._size = 0

    def _grow(self) -> None:
        new_data: List[Optional[Any]] = [None] * (len(self._data) * 2)
        for i in range(self._size):
            new_data[i] = self._data[(self._head + i) % len(self._data)]
        self._data = new_data
        self._head = 0
        self._tail = self._size

    def enqueue(self, item: Any) -> None:
        if self._size == len(self._data):
            self._grow()
        self._data[self._tail] = item
        self._tail = (self._tail + 1) % len(self._data)
        self._size += 1

    def dequeue(self) -> Any:
        if self._size == 0:
            raise IndexError("dequeue from empty queue")
        item = self._data[self._head]
        self._data[self._head] = None
        self._head = (self._head + 1) % len(self._data)
        self._size -= 1
        return item

    def is_empty(self) -> bool:
        return self._size == 0

    def __len__(self) -> int:
        return self._size


class HeapItem:
    def __init__(self, priority: float, value: Any):
        self.priority = priority
        self.value = value

    def __lt__(self, other: "HeapItem") -> bool:
        return self.priority < other.priority


class MinHeap:
    def __init__(self) -> None:
        self._data: List[HeapItem] = []

    def _sift_up(self, idx: int) -> None:
        while idx > 0:
            parent = (idx - 1) // 2
            if self._data[idx] < self._data[parent]:
                self._data[idx], self._data[parent] = self._data[parent], self._data[idx]
                idx = parent
            else:
                break

    def _sift_down(self, idx: int) -> None:
        n = len(self._data)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx
            if left < n and self._data[left] < self._data[smallest]:
                smallest = left
            if right < n and self._data[right] < self._data[smallest]:
                smallest = right
            if smallest == idx:
                break
            self._data[idx], self._data[smallest] = self._data[smallest], self._data[idx]
            idx = smallest

    def push(self, priority: float, value: Any) -> None:
        item = HeapItem(priority, value)
        self._data.append(item)
        self._sift_up(len(self._data) - 1)

    def pop(self) -> Any:
        if not self._data:
            raise IndexError("pop from empty heap")
        top = self._data[0]
        last = self._data.pop()
        if self._data:
            self._data[0] = last
            self._sift_down(0)
        return top.value

    def is_empty(self) -> bool:
        return not self._data


class PriorityQueue:
    """Thin wrapper around MinHeap used to schedule entity actions each year."""
    def __init__(self) -> None:
        self._heap = MinHeap()

    def push(self, priority: float, value: Any) -> None:
        self._heap.push(priority, value)

    def pop(self) -> Any:
        return self._heap.pop()

    def is_empty(self) -> bool:
        return self._heap.is_empty()


class BSTNode:
    def __init__(self, key: Any, value: Any):
        self.key = key
        self.value = value
        self.left: Optional["BSTNode"] = None
        self.right: Optional["BSTNode"] = None


class BinarySearchTree:
    """Stores final race populations so we can print them sorted by race name."""
    def __init__(self) -> None:
        self.root: Optional[BSTNode] = None

    def insert(self, key: Any, value: Any) -> None:
        self.root = self._insert(self.root, key, value)

    def _insert(self, node: Optional[BSTNode], key: Any, value: Any) -> BSTNode:
        if node is None:
            return BSTNode(key, value)
        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
        else:
            node.value = value
        return node

    def inorder(self) -> List[tuple]:
        out: List[tuple] = []
        self._inorder(self.root, out)
        return out

    def _inorder(self, node: Optional[BSTNode], out: List[tuple]) -> None:
        if node is None:
            return
        self._inorder(node.left, out)
        out.append((node.key, node.value))
        self._inorder(node.right, out)


# -----------------------------
# Domain objects: Race / Entity
# -----------------------------

class Race:
    def __init__(
        self,
        index: int,
        name: str,
        symbol: str,
        lifespan: int,
        reproduction_rate: float,
        fighting_ability: int,
        movement_predisposition: int,
    ):
        """
        reproduction_rate: expected number of new letters per existing letter per year.
            Example: Elves 0.1 => each E letter spawns a new E every 10 years on average.
        movement_predisposition: number of grid steps a letter attempts to move per year.
        """
        self.index = index
        self.name = name
        self.symbol = symbol
        self.lifespan = lifespan
        self.reproduction_rate = reproduction_rate
        self.fighting_ability = fighting_ability
        self.movement_predisposition = movement_predisposition


class Entity:
    _next_id = 1

    def __init__(self, race: Race, row: int, col: int):
        self.id = Entity._next_id
        Entity._next_id += 1
        self.race = race
        self.row = row  # 0 = bottom row
        self.col = col  # 0 = leftmost column
        self.age = 0
        self.alive = True


# -----------------------------
# Simulation helpers
# -----------------------------

def create_races() -> List[Race]:
    races: List[Race] = []
    # Index order matches the final population bar chart: Men, Dwarves, Elves, Orcs, Hobbits
    races.append(Race(0, "Men",     "M", lifespan=80,   reproduction_rate=8.0, fighting_ability=2, movement_predisposition=8))
    races.append(Race(1, "Dwarves", "D", lifespan=400,  reproduction_rate=1.0,  fighting_ability=4, movement_predisposition=1))
    races.append(Race(2, "Elves",   "E", lifespan=1500, reproduction_rate=0.1, fighting_ability=5, movement_predisposition=3))
    races.append(Race(3, "Orcs",    "O", lifespan=40,   reproduction_rate=12.0, fighting_ability=3, movement_predisposition=8))
    races.append(Race(4, "Hobbits", "H", lifespan=100,  reproduction_rate=2.0, fighting_ability=1, movement_predisposition=2))
    return races


def ask_initial_units(races: List[Race]) -> List[int]:
    print("Enter the initial number of units (in thousands) for each race.")
    print("For example, entering 12 for Dwarves means 12,000 dwarves, represented by 12 letters on the map.\n")
    counts: List[int] = []
    for race in races:
        while True:
            try:
                val = int(input(f"{race.name}: "))
                if val < 0:
                    raise ValueError()
                counts.append(val)
                break
            except ValueError:
                print("Please enter a non-negative integer.")
    return counts


def random_initial_placement(entities: LinkedList, races: List[Race], initial_counts: List[int]) -> List[List[Optional[Entity]]]:
    """Place starting entities randomly on the 7×12 board based on initial counts."""
    total_cells = ROWS * COLS
    total_letters = sum(initial_counts)
    if total_letters > total_cells:
        print(f"\nWarning: total letters ({total_letters}) exceed available cells ({total_cells}).")
        print("Only the first", total_cells, "letters will be placed on the map.\n")

    # List of all cell coordinates (row, col), row 0 = bottom
    all_cells = [(r, c) for r in range(ROWS) for c in range(COLS)]
    random.shuffle(all_cells)

    board: List[List[Optional[Entity]]] = [[None for _ in range(COLS)] for _ in range(ROWS)]
    cell_index = 0

    for race, count in zip(races, initial_counts):
        for _ in range(count):
            if cell_index >= len(all_cells):
                return board
            row, col = all_cells[cell_index]
            cell_index += 1
            entity = Entity(race, row, col)
            entities.append(entity)
            board[row][col] = entity

    return board


def move_entities_for_year(
    year: int,
    entities: LinkedList,
    races: List[Race],
    events: Stack,
) -> List[List[List[Entity]]]:
    """
    Move all alive entities for one year:
      - Age increments.
      - Old entities die.
      - Movement order determined by priority queue.
      - Returns a cell -> [entities] grid before reproduction, for battles.
    """
    # First age and put alive entities into a Queue
    q = Queue()
    for entity in entities:
        if not entity.alive:
            continue
        entity.age += 1
        if entity.age > entity.race.lifespan:
            entity.alive = False
            events.push(f"Year {year}: {entity.race.name} #{entity.id} died of old age at ({entity.col}, {entity.row}).")
        else:
            q.enqueue(entity)

    # Now schedule them into a priority queue
    pq = PriorityQueue()
    while not q.is_empty():
        e = q.dequeue()
        # smaller priority => earlier move; faster races tend to move earlier
        speed = e.race.movement_predisposition
        priority = 1.0 / (speed + 1) + random.random() * 0.1
        pq.push(priority, e)

    # Perform movement
    while not pq.is_empty():
        e = pq.pop()
        if not e.alive:
            continue
        steps = max(0, e.race.movement_predisposition)
        for _ in range(steps):
            dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])  # (row, col)
            new_row = min(max(e.row + dr, 0), ROWS - 1)
            new_col = min(max(e.col + dc, 0), COLS - 1)
            e.row, e.col = new_row, new_col

    # Build grid of cell -> list of entities (for battles)
    cell_lists: List[List[List[Entity]]] = [[[] for _ in range(COLS)] for _ in range(ROWS)]
    for e in entities:
        if not e.alive:
            continue
        cell_lists[e.row][e.col].append(e)

    return cell_lists


def resolve_battles(year: int, cell_lists: List[List[List[Entity]]], events: Stack) -> None:
    """If multiple races occupy the same tile, only the strongest race survives."""
    for r in range(ROWS):
        for c in range(COLS):
            if len(cell_lists[r][c]) <= 1:
                continue
            occupants = cell_lists[r][c]
            # Determine strongest race
            best_strength = max(e.race.fighting_ability for e in occupants)
            strongest = [e for e in occupants if e.race.fighting_ability == best_strength]
            winner = random.choice(strongest)
            for e in occupants:
                if e is winner:
                    continue
                e.alive = False
                events.push(
                    f"Year {year}: {winner.race.name} #{winner.id} defeated {e.race.name} #{e.id} at ({c}, {r})."
                )
            # After battle, only winner remains in this cell
            cell_lists[r][c] = [winner]


def rebuild_entities_and_grid(
    entities: LinkedList,
) -> List[List[Optional[Entity]]]:
    """Remove dead entities from the list and rebuild a single-entity-per-cell board."""
    entities.remove_if(lambda e: not e.alive)

    board: List[List[Optional[Entity]]] = [[None for _ in range(COLS)] for _ in range(ROWS)]
    for e in entities:
        if not e.alive:
            continue
        board[e.row][e.col] = e
    return board


def reproduction_step(
    year: int,
    entities: LinkedList,
    races: List[Race],
    board: List[List[Optional[Entity]]],
    repro_acc: np.ndarray,
    events: Stack,
) -> None:
    """Perform reproduction for each race, adding new letters in random empty cells."""
    # Count letters per race
    counts = [0] * len(races)
    for e in entities:
        if not e.alive:
            continue
        counts[e.race.index] += 1

    # List empty cells
    empty_cells = [(r, c) for r in range(ROWS) for c in range(COLS) if board[r][c] is None]
    random.shuffle(empty_cells)

    for race in races:
        idx = race.index
        num_letters = counts[idx]
        if num_letters == 0:
            continue
        repro_acc[idx] += race.reproduction_rate * num_letters
        new_letters = int(repro_acc[idx])
        if new_letters <= 0:
            continue
        repro_acc[idx] -= new_letters

        for _ in range(new_letters):
            if not empty_cells:
                return  # board is full, no more reproduction possible
            r, c = empty_cells.pop()
            child = Entity(race, r, c)
            entities.append(child)
            board[r][c] = child
            events.push(
                f"Year {year}: {race.name} reproduced -> new {race.name} #{child.id} at ({c}, {r})."
            )


def compute_population_counts(entities: LinkedList, races: List[Race]) -> List[int]:
    counts = [0] * len(races)
    for e in entities:
        if not e.alive:
            continue
        counts[e.race.index] += 1
    return counts


# -----------------------------
# Display helpers
# -----------------------------

def print_board(board: List[List[Optional[Entity]]]) -> None:
    """Print the 7×12 Middle Earth map using the given board layout."""
    border = "-" * (4 * COLS + 1)
    for row in range(ROWS - 1, -1, -1):
        print(border)
        line = "|"
        for col in range(COLS):
            e = board[row][col]
            ch = e.race.symbol if e is not None else " "
            line += f" {ch} |"
        print(line)
    print(border)


def print_population_bar_chart(counts: List[int], races: List[Race]) -> None:
    """Print a vertical bar chart (0..10) of populations by race."""
    print("\nFinal population levels (each bar scaled 0–10):\n")
    max_count = max(counts) if counts else 1
    if max_count == 0:
        print("All races are extinct.")
        return

    # Scale each population to a height from 0 to 10
    heights = []
    for c in counts:
        if c <= 0:
            heights.append(0)
        else:
            h = int(round(10.0 * c / max_count))
            if h == 0:
                h = 1
            heights.append(h)

    # Print from top (10) to bottom (1)
    for level in range(10, 0, -1):
        row_str = f"{level:2d} |"
        for h in heights:
            if h >= level:
                row_str += "   |||   "
            else:
                row_str += "         "
        print(row_str)

    # X-axis
    axis = "0  |"
    for _ in heights:
        axis += "---|||---"
    print(axis)
    # Labels
    label_line = "     "
    for race in races:
        # pad to roughly align with bars
        label = race.name
        label_line += f" {label:8}"
    print(label_line)
    print()


def final_summary(year: int, entities: LinkedList, races: List[Race]) -> None:
    print("\n===== FINAL SUMMARY =====\n")
    counts = compute_population_counts(entities, races)

    # Store in BST keyed by race name
    bst = BinarySearchTree()
    for race, count in zip(races, counts):
        bst.insert(race.name, count)

    print("Final populations by race (sorted by name):")
    for name, count in bst.inorder():
        print(f"  {name:8}: {count:3d} letters on the map")
    print()

    print_population_bar_chart(counts, races)


# -----------------------------
# Main simulation loop
# -----------------------------

def run_simulation() -> None:
    cls()
    print("**********************************************")
    print("          Middle Earth Simulation")
    print("**********************************************\n")

    races = create_races()
    initial_counts = ask_initial_units(races)

    while True:
        try:
            max_years = int(input("\nEnter maximum number of years to simulate: "))
            if max_years <= 0:
                raise ValueError()
            break
        except ValueError:
            print("Please enter a positive integer.")

    entities = LinkedList()
    board = random_initial_placement(entities, races, initial_counts)

    # NumPy array to record populations (rows=races, columns=years)
    pop_history = np.zeros((len(races), max_years + 1), dtype=int)
    repro_acc = np.zeros(len(races), dtype=float)

    # Initial population record (year 0)
    counts0 = compute_population_counts(entities, races)
    for i, c in enumerate(counts0):
        pop_history[i, 0] = c

    year = 0
    while year < max_years:
        year += 1
        cls()
        print(f"Year {year}")
        print_board(board)

        events = Stack()
        # Movement & aging
        cell_lists = move_entities_for_year(year, entities, races, events)
        # Battles
        resolve_battles(year, cell_lists, events)
        # Rebuild clean single-entity board
        board = rebuild_entities_and_grid(entities)
        # Reproduction
        reproduction_step(year, entities, races, board, repro_acc, events)

        # Record populations
        counts = compute_population_counts(entities, races)
        for i, c in enumerate(counts):
            pop_history[i, year] = c

        # Display simple per-year statistics
        print("\nPopulation counts:")
        for race, c in zip(races, counts):
            print(f"  {race.name:8}: {c:3d} letters")

        if events.is_empty():
            print("\nNo major events this year.")
        else:
            print("\nEvents this year (most recent first):")
            while not events.is_empty():
                print("  -", events.pop())

        # Stop if all extinct
        if sum(counts) == 0:
            print("\nAll races are extinct. Ending simulation.")
            break

        # Allow user to stop early
        choice = input("\nPress Enter to advance to next year, or type 'q' then Enter to stop: ").strip().lower()
        if choice == "q":
            break

    final_summary(year, entities, races)


if __name__ == "__main__":
    run_simulation()

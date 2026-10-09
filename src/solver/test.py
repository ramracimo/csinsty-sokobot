from sokobot import Solver

def main():
    layout = ["#######",
            "#@ $$ #",
            "#     #",
            "#######"]

    walls, crates, goals = set(), set(), {(1, 5)}
    player = None

    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            if ch == '#':
                walls.add((r, c))
            elif ch == '$':
                crates.add((r, c))
            elif ch == '@':
                player = (r, c)

    solver = Solver(len(layout[0]), len(layout), walls, goals)
    dist = solver.movableRegion(player, frozenset(crates))

    for r in range(len(layout)):
        line = ""
        for c in range(len(layout[0])):
            if (r, c) in dist:
                line += str(dist[(r, c)])
            else:
                line += layout[r][c]
        print(line)

if __name__ == "__main__":
    main()
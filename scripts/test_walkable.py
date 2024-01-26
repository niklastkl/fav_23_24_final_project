import numpy as np
import matplotlib.pyplot as plt


def is_walkable(p0, p1, occupancy_matrix):
    traversed_fields = []
    x1 = p0[0]
    y1 = p0[1]
    x2 = p1[0]
    y2 = p1[1]
    #i = np.nan            # loop counter
    ystep = np.nan
    xstep = np.nan    # the step on y and x axis
    error = np.nan           # the error accumulated during the increment
    errorprev = np.nan     # *vision the previous value of the error variable
    y = y1
    x = x1  # the line points
    ddy = np.nan
    ddx = np.nan        # compulsory variables: the double values of dy and dx
    dx = x2 - x1;
    dy = y2 - y1;
    traversed_fields.append((y1, x1));  # first point
                              # NB the last point can't be here, because of its previous point (which has to be verified)
    if (dy < 0):
        ystep = -1
        dy = -dy
    else:
        ystep = 1
    if (dx < 0):
        xstep = -1
        dx = -dx
    else:
        xstep = 1
    ddy = 2 * dy  # work with double values for full precision
    ddx = 2 * dx
    if (ddx >= ddy):  # first octant (0 <= slope <= 1)
        # compulsory initialization (even for errorprev, needed when dx==dy)
        error = dx
        errorprev = dx  # start in the middle of the square
        for i in range(dx):  # do not use the first point (already done)
            x += xstep
            error += ddy
            if (error > ddx):  # increment y if AFTER the middle ( > )
                y += ystep
                error -= ddx
                # three cases (octant == right->right-top for directions below):
                if (error + errorprev < ddx):  # bottom square also
                    traversed_fields.append((y-ystep, x))
                elif (error + errorprev > ddx):  # left square also
                    traversed_fields.append((y, x-xstep))
                else:  # corner: bottom and left squares also
                    traversed_fields.append((y-ystep, x))
                    traversed_fields.append((y, x-xstep))
            traversed_fields.append((y, x))
            errorprev = error
    else: # the same as above
        error = dy
        errorprev = dy
        for i in range(dy):
            y += ystep;
            error += ddx;
            if (error > ddy):
                x += xstep;
                error -= ddy;
                if (error + errorprev < ddy):
                    traversed_fields.append((y, x-xstep))
                elif (error + errorprev > ddy):
                    traversed_fields.append((y-ystep, x))
                else:
                    traversed_fields.append((y, x-xstep))
                    traversed_fields.append((y-ystep, x))
            traversed_fields.append((y, x))
            errorprev = error

    if (not y == y2) | (not x == x2):  # the last point (y2,x2) has to be the same with the last point of the algorithm
        print("Algorithm did not end at endpoint!")
        return False, traversed_fields
    for field in traversed_fields:
        if occupancy_matrix[field[0], field[1]] > 0:
            return False, traversed_fields
    return True, traversed_fields

"""def is_walkable(p0, p1, occupancy_matrix):
    # https://www.gamedeveloper.com/programming/toward-more-realistic-pathfinding
    #a = p1
    #p1 = p0
    #p0 = a
    traversed_fields = []
    n_horz_intersects = abs(p1[1] - p0[1])
    horz_intersects = abs(float(p1[0] - p0[0])) / float(abs(p1[1] - p0[1]) + 1)
    dir_horz = int((p1[0] - p0[0]) / max(abs(float(p1[0] - p0[0])), 1e-6))
    dir_vert = int((p1[1] - p0[1]) / max(abs(float(p1[1] - p0[1])), 1e-6))
    for i in range(n_horz_intersects + 1):
        traversed_fields = traversed_fields + [(j, p0[1] + dir_vert * i) for j in
                                               range(p0[0] + dir_horz*int(i * horz_intersects + 0.5), p0[0] + dir_horz * int((i + 1) * horz_intersects + 0.5) + 1)]
    for field in traversed_fields:
        if occupancy_matrix[field[1], field[0]] > 0:
            return False
    return True, traversed_fields
"""
def main():
    height = 50
    width = 75
    n_lines = 5
    n_boxes = 200
    np.random.seed(1)
    grid = np.zeros((height, width), dtype=np.uint8)
    plt.figure()
    for i in range(n_lines):
        p0 = np.random.randint([0, 0], [width, height])
        p1 = np.random.randint([0, 0], [width, height])
        feasible, traversed_fields = is_walkable(p0, p1, grid)
        for field in traversed_fields:
            grid[field[0], field[1]] = 255
        if feasible:
            color = "g"
        else:
            color = "r"
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], c=color)
    plt.imshow(grid)
    plt.show()

    grid = np.zeros((height, width), dtype=np.uint8)
    idxs_occupied_height = np.random.randint(0, height, n_boxes)
    idxs_occupied_width = np.random.randint(0, width, n_boxes)
    grid[idxs_occupied_height, idxs_occupied_width] = 255
    plt.figure()
    plt.imshow(grid)
    for i in range(n_lines):
        p0 = np.random.randint([0, 0], [width, height])
        p1 = np.random.randint([0, 0], [width, height])
        feasible, traversed_fields = is_walkable(p0, p1, grid)
        if feasible:
            color = "g"
        else:
            color = "r"
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], c=color)
    plt.show()

if __name__ == "__main__":
    main()
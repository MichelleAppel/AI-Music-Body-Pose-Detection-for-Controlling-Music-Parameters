import math

def euclidean_distance(p1, p2):
    """
    Calculate the Euclidean distance between two n-dimensional points.

    Args:
        p1: A tuple or list representing the coordinates of the first point.
        p2: A tuple or list representing the coordinates of the second point.

    Returns:
        The Euclidean distance between the two points.

    Raises:
        ValueError: If the two input points have different dimensions.

    """
    # Check if the two input points have the same number of dimensions
    if len(p1) != len(p2):
        raise ValueError("The input points must have the same number of dimensions.")

    # Calculate the squared distance between the two points for each dimension
    squared_distance = [(a - b) ** 2 for a, b in zip(p1, p2)]

    # Take the square root of the sum of the squared distances to get the Euclidean distance
    distance = math.sqrt(sum(squared_distance))

    # Return the Euclidean distance
    return distance

def isBadVersion():
    pass

def firstBadVersion(self, n: int) -> int:
    left = 1
    right = 0

    while left < right:
        mid = (left + right) // 2
        if isBadVersion(mid):
            right = mid
        else:
            left = mid + 1

    return right

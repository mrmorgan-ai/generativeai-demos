"""
Unique Binary Search Trees - Multiple Solutions
Problem: Given n, return number of structurally unique BSTs with nodes 1 to n.
"""

# Solution 1: Bottom-up Dynamic Programming
# Time: O(n²), Space: O(n)
def numTrees_dp(n):
    """
    Bottom-up DP approach.
    dp[i] = number of unique BSTs with i nodes
    """
    if n <= 1:
        return 1
    
    # dp[i] represents number of unique BSTs with i nodes
    dp = [0] * (n + 1)
    dp[0] = 1  # Empty tree
    dp[1] = 1  # Single node
    
    # Fill dp table for 2 to n nodes
    for nodes in range(2, n + 1):
        for root in range(1, nodes + 1):
            left_nodes = root - 1      # Nodes in left subtree
            right_nodes = nodes - root # Nodes in right subtree
            dp[nodes] += dp[left_nodes] * dp[right_nodes]
    
    return dp[n]


# Solution 2: Mathematical Formula (Catalan Numbers)
# Time: O(n), Space: O(1) - BEST SOLUTION
def numTrees_catalan(n):
    """
    Direct calculation using Catalan number formula.
    C(n) = (1/(n+1)) * C(2n, n) where C(2n, n) is binomial coefficient.
    This is the most efficient solution.
    """
    if n <= 1:
        return 1
    
    # Calculate C(2n, n) / (n+1) = (2n)! / ((n+1)! * n!)
    # We'll calculate this iteratively to avoid large factorials
    result = 1
    for i in range(n):
        result = result * (2 * n - i) // (i + 1)
    
    return result // (n + 1)


# Solution 3: Optimized Mathematical (Alternative)
# Time: O(n), Space: O(1)
def numTrees_catalan_v2(n):
    """
    Another way to calculate Catalan numbers iteratively.
    Uses the recurrence: C(n) = C(n-1) * (4*n - 2) / (n + 1)
    """
    if n <= 1:
        return 1
    
    catalan = 1
    for i in range(2, n + 1):
        catalan = catalan * (4 * i - 2) // (i + 1)
    
    return catalan


# Solution 4: Recursive with Memoization (for reference)
# Time: O(n²), Space: O(n)
def numTrees_memo(n, memo=None):
    """
    Top-down approach with memoization.
    Less efficient than bottom-up but shows the recursive structure clearly.
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return 1
    
    result = 0
    for root in range(1, n + 1):
        left_nodes = root - 1
        right_nodes = n - root
        left_ways = numTrees_memo(left_nodes, memo)
        right_ways = numTrees_memo(right_nodes, memo)
        result += left_ways * right_ways
    
    memo[n] = result
    return result


# Test function to verify all solutions
def test_solutions():
    """Test all solutions with known results."""
    test_cases = [
        (0, 1),   # Empty tree
        (1, 1),   # Single node
        (2, 2),   # Two structures
        (3, 5),   # Five structures  
        (4, 14),  # Fourteen structures
        (5, 42),  # Forty-two structures
    ]
    
    solutions = [
        ("DP", numTrees_dp),
        ("Catalan v1", numTrees_catalan),
        ("Catalan v2", numTrees_catalan_v2),
        ("Memoization", numTrees_memo)
    ]
    
    print("Testing all solutions:")
    print("n\tExpected\t" + "\t".join([name for name, _ in solutions]))
    print("-" * 60)
    
    for n, expected in test_cases:
        results = []
        for name, func in solutions:
            result = func(n)
            results.append(str(result))
        
        print(f"{n}\t{expected}\t\t" + "\t\t".join(results))
        
        # Verify all solutions match
        if all(int(r) == expected for r in results):
            print(f"✓ All solutions correct for n={n}")
        else:
            print(f"✗ Mismatch for n={n}")
        print()


# Performance comparison
def performance_test():
    """Compare performance of different approaches."""
    import time
    
    n = 19  # Large enough to see differences
    
    solutions = [
        ("Bottom-up DP", numTrees_dp),
        ("Catalan Formula v1", numTrees_catalan),
        ("Catalan Formula v2", numTrees_catalan_v2),
    ]
    
    print(f"Performance test with n={n}:")
    print("Method\t\t\tTime (seconds)\tResult")
    print("-" * 50)
    
    for name, func in solutions:
        start = time.time()
        result = func(n)
        elapsed = time.time() - start
        print(f"{name:<20}\t{elapsed:.6f}\t{result}")


if __name__ == "__main__":
    # Run tests
    test_solutions()
    print("\n" + "="*60 + "\n")
    performance_test()
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("Use numTrees_catalan() - O(n) time, O(1) space")
    print("It's the most efficient solution using Catalan number formula.")

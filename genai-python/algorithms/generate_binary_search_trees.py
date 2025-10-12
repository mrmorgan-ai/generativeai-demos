from typing import List, Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        def generate_trees(start, end):
            if start > end:
                return [None,]
                
            list_bst = []
            for root in range(start, end+1):
                left_bst = generate_trees(start,root-1)
                right_bst = generate_trees(root+1,end)
                
                for lbst in left_bst:
                    for rbst in right_bst:
                        bst = TreeNode(root)
                        bst.left = lbst
                        bst.right = rbst
                        list_bst.append(bst)
            return list_bst
        return generate_trees(1,n) if n else []

    # Using dynamic programming
    def dp_trees(n):
        dp = [[[None] for i in range(n+2)] for j in range(n+2)]
        print (dp)
        for i in range(n, 0, -1):
            for j in range(i, n+1):
                res = []
                for k in range(i, j+1):
                    leftSubtrees = dp[i][k-1]
                    rightSubtrees = dp[k+1][j]
                    
                    for left in leftSubtrees:
                        for right in rightSubtrees:
                            root = TreeNode(k, left, right)
                            res.append(root)

                dp[i][j] = res

test = Solution()

test.dp_trees(3)

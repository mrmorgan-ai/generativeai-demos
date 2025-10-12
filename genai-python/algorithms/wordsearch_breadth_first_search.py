class Solution:
    def exist(self, board, word):
        # Recursive Function
        def backtrack(i, j, k):
            """
            Function that explore all board searching for words
            (i,j) = current board position
            k = index of the word
            """
            if k == len(word):
                return True
            # Condition to cancel recursive
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
                return False
            
            # Temp is the cell visited
            temp = board[i][j]
            # Here we declare visited cell as empty to avoid duplications
            board[i][j] = ''
            
            # To pass this conditional any conditions must be True
            if backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1):
                return True
            
            # Reseth the cell to original value if none of recursive is True
            board[i][j] = temp
            # Return False because word wasn't found
            return False
        
        # Iterate the recursive all the directions for a given position
        # Return True if 
        for i in range(len(board)):
            for j in range(len(board[0])):
                if backtrack(i, j, 0):
                    return True
        return False

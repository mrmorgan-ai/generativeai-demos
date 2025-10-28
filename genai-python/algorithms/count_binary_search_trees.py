import time
def count_unique_bst_recursive(n: int):
    begin = time.time()
    # Base cases
    if n <= 1:
        return 1
    
    # Initialize accumulator
    total_bst = 0
    
    for root in range(1, n+1):
        #print(f"Root: {root}")
        left_nodes = root - 1
        right_nodes = n - root
        #print(f"Left nodes: {left_nodes}, Right nodes: {right_nodes}")
        
        left_bst = count_unique_bst_recursive(left_nodes)
        right_bst = count_unique_bst_recursive(right_nodes)
        #print(f"Left BST: {left_bst}, Right BST: {right_bst}")
        
        partial_bst = left_bst*right_bst
        #print(f"Partial BST for root {root}: {partial_bst}")
        total_bst += partial_bst
        #print(f"Total BST so far: {total_bst}\n###############################\n")
    finish = time.time()
    elapse = finish - begin
    print(f"Time elapsed: {elapse}")
    return total_bst


print(count_unique_bst_recursive(5))


def count_unique_bst_catalan(n: int):
    if n <= 1:
        return 1
    
    result = 1
    for i in range(n):
        result = result*(2*n - i)//(i+1)
    
    finish = time.time()
    return result//(n+1)

print(count_unique_bst_catalan(5))
        
    

from TreeForms import *

arr = np.ones(3, dtype=np.int8)

CN = TreeForm(CN_OPERATIONS[1:], 10)
node = CN.new_node(10)
print(node)
c = 1
while node.iterate(arr):
    print(node)
    c += 1
print(c)
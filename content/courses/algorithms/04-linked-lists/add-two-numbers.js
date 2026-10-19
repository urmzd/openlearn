#!/usr/bin/env node

/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} l1
 * @param {ListNode} l2
 * @return {ListNode}
 */
var addTwoNumbers = function(l1, l2) {
    return helper(l1, l2)
};

// Traverse through the list and sum the values (O(max(n, m)))
// Construct a new list containing the sums for l1 and l2

// Add value, go to next for each 
// helper can also traverse backwards making this an O(n) process
// Return the reversed head
function isNullish(x){
    return x === undefined || x === null
}

function areNullish(...args) {
    return args?.every(isNullish)
}

function helper(node1, node2, head=null, tail=new ListNode(), carry=0, base=10) {
    // Extract the numbers
    const val1 = node1?.val ?? 0
    const val2 = node2?.val ?? 0
    
    // Get the remainder + carry
    const val = (val1 + val2 + carry) % base
    const carry2 = Math.floor((val1 + val2 + carry) / base)
    
    // Create new node
        
    const node = new ListNode(val)
    tail.next = node
    
    // Point head if null
    let _head = isNullish(head) ? node : head;
    console.log(_head)
    
    // Base case
    console.log(val, carry2, areNullish(node1?.next, node?.next))
    if (carry2 === 0 && areNullish(node1?.next, node2?.next)) {
        return _head
    }
    
    // Continue traversing
    return helper(node1?.next, node2?.next, _head, node, carry2)
}

# Huffman coding

This Java program uses a Huffman tree to encode a text file and decode it again. It is the full-scale version of the Huffman pattern described in [Greedy algorithms](../README.md): count how often each character appears, repeatedly merge the two least probable subtrees, and read each character's code off its path from the root.

The program is split across four files. [`Pair.java`](Pair.java) holds a character and its probability (its frequency divided by the number of characters in the message) and orders pairs by probability. [`BinaryTree.java`](BinaryTree.java) is a generic binary tree with pre-order, in-order, and post-order traversals. [`Huffman.java`](Huffman.java) does the work. Its `encode` method reads the message, removes whitespace, builds the sorted table of character probabilities, and then builds the tree with the two-queue method instead of a heap: the leaves start in one queue in sorted order, merged subtrees go into a second queue, and each step takes the two smallest items from the fronts of the two queues. Because merged weights are produced in non-decreasing order, both queues stay sorted, so the smallest item is always at one of the two fronts and no heap is needed. `findEncoding` then walks the tree to assign each character its bit string, and `encode` writes the dictionary table (symbol, probability, code) to `Huffman.txt` and the encoded message to `Encoded.txt`. The `decode` method reads the encoded file and the dictionary back and maps each code to its character. [`HuffmanDemo.java`](HuffmanDemo.java) runs encoding and then decoding.

To run it:

1. Compile the demo with `javac HuffmanDemo.java`, which compiles the other files as well.
2. Run it with `java HuffmanDemo`.
3. When prompted, enter, in order, the name of the message file to encode, then `Encoded.txt` and `Huffman.txt`, the encoded message and dictionary the encoder just wrote, which the decoder reads back.

When it finishes, the dictionary file contains the table of Huffman codes, and the program reports that the file was encoded and decoded successfully.

## Key terms

- **Huffman tree**: the binary tree Huffman's algorithm builds, with each character at a leaf; a character's code is its path from the root, 0 for a left branch and 1 for a right one, so frequent characters sit near the root and get short codes.
- **Two-queue method**: building a Huffman tree from leaves sorted by weight without a heap: leaves wait in one queue, merged subtrees join a second, and each step merges the two smallest items found at the two fronts. Merged weights come out in non-decreasing order, so both queues stay sorted and the build after sorting is $O(k)$.
- **Huffman code table**: the dictionary from each character to its bit string, read off the Huffman tree; the decoder needs it, or the tree, to turn the bit stream back into text.

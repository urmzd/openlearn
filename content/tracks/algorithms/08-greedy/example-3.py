def greedy_algorithm(sequence):
    sequence_enumerated = list(enumerate(sequence))
    sequence_sorted = sorted(sequence_enumerated, key=lambda x: x[1])

    sub_seqs = []
    curr_sub_seq = []

    for key, value in sequence_sorted:
        # Add the first available value to our
        # current subsequence.
        if not curr_sub_seq:
            curr_sub_seq.append((key, value))
        else:
            _key, _value = curr_sub_seq[-1]

            # If constraints are satisfied, add
            # value to our subsequence.
            if _key < key and _value - 3 <= value:
                curr_sub_seq.append((key, value))
            else:
                # Store our current subsequence and restart counter.
                sub_seqs.append([x[1] for x in curr_sub_seq])
                curr_sub_seq = [(key, value)]

    # Add subsequence determined during last iteration.
    sub_seqs.append([x[1] for x in curr_sub_seq])
    return sub_seqs

#include <stdio.h>

#define MIN(a, b) (a < b ? a : b)

int optBST(int *F, int n)
{
    // The cost-table
    int A[n][n];

    // The root-table
    int R[n][n];

    // Calculate in O(n^2) the probability of all the keys for each A[i][j].
    int P[n][n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = i; k <= j; k++)
            {
                P[i][j] += F[k];
            }
        }
    }

    // Initialize the cost-table so that the first diagonal is the given frequencies.
    for (int i = 0; i < n; i++)
    {
        A[i][i] = F[i];
        R[i][i] = i;
    }

    // Traverse the cost-table diagonally.
    for (int size = 2; size <= n; size++)
    {
        for (int i = 0; i < n - size + 1; i++)
        {
            int j = i + size - 1;

            // Set the minimum cost initially to the cost of the cell below the current cell.
            A[i][j] = A[i + 1][j];
            // Set our root to the cell below it.
            R[i][j] = i + 1;

            // Traverse diagonally
            // Reducing the range of this loop is how we get from O(n^3) to O(n^2).
            // Use a smaller range i.e. Knuth's Lemma
            // Compare our current minimum cost with the diagonal cells and replace it if it is smaller.
            for (int r = R[i][j - 1]; r <= R[i + 1][j]; r++)
            {
                A[i][j] = MIN(A[i][j], A[i][r - 1] + A[r + 1][j]);
                if (A[i][j] == A[i][r - 1] + A[r + 1][j])
                {
                    // Set our root to the cell at this diagonal of it.
                    R[i][j] = r;
                }
            }

            // Compare our current minimum cost with the cost of the cell to the left and replace if it is smaller.
            A[i][j] = MIN(A[i][j], A[i][j - 1]);
            if (A[i][j] == A[i][j - 1])
            {
                // Set our root to the cell at the left of it.
                R[i][j] = j - 1;
            }

            A[i][j] += P[i][j];
        }
    }

    fprintf(stderr, "\t");
    for (int j = 0; j < n; j++)
    {
        fprintf(stderr, "\t%i", j);
    }
    fprintf(stderr, "\n");

    for (int i = 0; i < n; i++)
    {
        fprintf(stderr, "\t%i", i);
        for (int j = 0; j < n; j++)
        {
            fprintf(stderr, "\t");
            if (j >= i)
            {
                fprintf(stderr, "%i", A[i][j]);
            }
        }

        fprintf(stderr, "\n");
    }

    // Return the top right corner's cost (i.e. optimal cost).
    return (A[0][n - 1]);
}

int main() {
    int F[] = { 5, 3, 4, 1, 1, 2, 1 };
    printf("%i\n", optBST(F, 7));
    return 0;
}

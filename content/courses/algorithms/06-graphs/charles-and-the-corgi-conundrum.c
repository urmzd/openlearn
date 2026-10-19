/**
 * PROBLEM: Charles and the Corgi Conundrum
 * * DESCRIPTION:
 * Charles manages a pond of 'n' corgis. Some corgis have formed mutual social 
 * bonds. Social influence in the pond is transitive: when you interact with 
 * a corgi, you aren't just interacting with them, but everyone in their 
 * immediate social circle.
 * * THE GOAL:
 * Calculate the average "Social Reach Score" across all 'n' corgis in the pond.
 * * RULES FOR SOCIAL REACH:
 * 1. Every corgi starts with a base reach score of 1 (itself).
 * 2. For every direct friend a corgi has, add that friend's total number 
 * of bonds to the corgi's reach score.
 * 3. The final answer is the sum of all individual reach scores divided 
 * by the total number of corgis (n).
 * * INPUT FORMAT:
 * - Line 1: 's' (number of ponds/test cases).
 * - For each pond:
 * - Line 1: 'n' (number of corgis) and 'm' (number of bonds).
 * - Next 'm' lines: integers 'u' and 'v' representing a mutual bond.
 * * OUTPUT FORMAT:
 * - "Pond #x: [average]" (formatted to 3 decimal places).
 * * SAMPLE INPUT:
 * 1
 * 3 2
 * 1 2
 * 2 3
 * * SAMPLE OUTPUT:
 * Pond #1: 3.000
 * * EXPLANATION:
 * Corgi 1: Base(1) + Friend2_Bonds(2) = 3
 * Corgi 2: Base(1) + Friend1_Bonds(1) + Friend3_Bonds(1) = 3
 * Corgi 3: Base(1) + Friend2_Bonds(2) = 3
 * Average: (3 + 3 + 3) / 3 = 3.000
 */

#include <stdio.h>

// TODO: Implement the solution logic here.

int main() {

    

    // Your code starts here...
    return 0;
}

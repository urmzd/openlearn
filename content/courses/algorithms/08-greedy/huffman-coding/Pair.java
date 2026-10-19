/**
 * The Pair class is used to hold on to a character and it's respective probability (frequency in a message / total # of ASCII characters in message).
 * It is used primarly in the Huffman program.
 * @author Urmzd Mukhammadnaim
 */
public class Pair implements Comparable<Pair>{

    // Attributes.
    private char value;
    private double prob;

    // Constructors.
    public Pair(){}

    public Pair(char value, double prob){
        this.value = value;
        this.prob = prob;
    }

    // Accessors and Mutators.
    public char getValue() {
        return value;
    }

    public void setValue(char value) {
        this.value = value;
    }

    public double getProb() {
        return prob;
    }

    public void setProb(double prob) {
        this.prob = prob;
    }

    // Returns 0 if same, > 0 if this is greater than that, and <0 if that is greater than this.
    @Override
    public int compareTo(Pair that){
        return Double.compare(this.prob, that.getProb());
    }

    // ToString method outputs the characters value and probability.
    @Override
    public String toString() {
        return String.format("Char: (%s), Prob: %.4f", value, prob);
    }

    
    
}
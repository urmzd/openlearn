// Imports.
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

/**
 * The Huffman Program uses the Greedy Algorithm (through Huffman coding) to
 * compress a file. It is able to encode a file given by the user and creates a
 * dictionary alongside the encoded file. Additionally, it is able to decode a
 * file given its dictionary.
 * 
 * @author Urmzd Mukhammadnaim
 */
public class Huffman {

    /**
     * The encode method reads in a file, compresses it using the Greedy Algorithm
     * and
     * 
     * @throws IOException Throws an exception if the file indicated by the user
     *                     does not exist.
     */
    public static void encode() throws IOException {

        String text = readText(); // Read in file.
        char[] textChars = text.replaceAll("\\s", "").toCharArray(); // Remove whitespace, new line characters and tabs.

        ArrayList<Pair> pairs = getSortedPairs(mapFreq(textChars), textChars.length); // Create a table with characters
                                                                                      // and their probabilities.

        ArrayList<BinaryTree<Pair>> S = new ArrayList<>(); // Create Queue1.
        ArrayList<BinaryTree<Pair>> T = new ArrayList<>(); // Create Queue2.

        BinaryTree<Pair> bt; // Create tree.

        // Add all characters and their corresponding probabilities as a binary-tree to
        // Queue1.
        for (Pair pair : pairs) {
            bt = new BinaryTree<>();
            bt.makeRoot(pair);
            S.add(bt);
        }

        // Allocate memory to variables.
        BinaryTree<Pair> A;
        BinaryTree<Pair> B;

        Pair p;
        BinaryTree<Pair> P;

        // Combine minimums of Queue1 and Queue2 until Queue1 is empty and Queue2's size
        // is 1.
        while (!(S.isEmpty() && T.size() == 1)) {

            // A and B are the minimus of Queue1 and Queue2 respectively.
            A = findMin(S, T);
            B = findMin(S, T);

            // Combine A and B and add them as a binary tree to Queue2.
            P = new BinaryTree<>();
            p = new Pair('\0', A.getData().getProb() + B.getData().getProb());
            P.makeRoot(p);
            P.attachLeft(A);
            P.attachRight(B);
            T.add(P);
        }

        HashMap<Pair, String> encodingMap = findEncoding(T.get(0)); // Create encoding scheme using the parent tree of
                                                                    // the Huffman tree created,
        HashMap<Character, String> charMap = new HashMap<>();

        String encoded = "";

        // Print Huffman dictionary to a file called 'Huffman.txt'.
        PrintWriter output = new PrintWriter("Huffman.txt");
        output.println("Symbol\t Prob\t HuffmanCode");

        for (Map.Entry<Pair, String> entry : encodingMap.entrySet()) {
            output.printf("%s\t %.4f\t %s\n", entry.getKey().getValue(), entry.getKey().getProb(), entry.getValue());
            charMap.put(entry.getKey().getValue(), entry.getValue()); // Store dictionary information.
        }

        output.close();

        // Using dictionary information, convert letters into binary and output it.
        for (int index = 0; index < text.length(); index++) {

            if (charMap.containsKey(text.charAt(index))) {
                encoded += charMap.get(text.charAt(index));
            } else {
                encoded += text.charAt(index);
            }
        }

        output = new PrintWriter("Encoded.txt");
        output.print(encoded);

        output.close();
    }

    /**
     * The decode() methods decodes an encoded file using its dictionary.
     * 
     * @throws IOException
     */
    public static void decode() throws IOException {
        // Read in encoded text and its dictionary.
        String encodedText = readText();
        Scanner dictionary = new Scanner(new File(getConsole().nextLine()));

        char c;
        String s;

        HashMap<String, Character> dictionaryMap = new HashMap<>();

        dictionary.nextLine(); // Discard table headers.

        // Add the respective binary and its character to the hashmap.
        while (dictionary.hasNext()) {
            c = dictionary.next().charAt(0);
            dictionary.next();
            s = dictionary.next();
            dictionaryMap.put(s, c);
        }

        String decoded = "";

        String binary = "";

        // Convert binary to letters using encoding schema.
        for (int index = 0; index < encodedText.length(); index++) {

            // If whitespace, new lines or tabs are found. Add it to the decoded text and
            // finish iteration.
            if (encodedText.charAt(index) != '0' && encodedText.charAt(index) != '1') {
                decoded += encodedText.charAt(index);
                continue;
            }

            // Else add encoded text to our temporary string binary.
            binary += encodedText.charAt(index);

            // If binary string is found in dictionary, convert it to a character and reset
            // binary string.
            if (dictionaryMap.containsKey(binary)) {
                decoded += dictionaryMap.get(binary);
                binary = "";
            }

        }

        // Output decoded message.
        PrintWriter output = new PrintWriter("Decoded.txt");

        output.println(decoded);
        output.close();
        dictionary.close();
        getConsole().close();
    }

    // Return map with a list of pairs with their respective proabilities.
    public static HashMap<Pair, String> findEncoding(BinaryTree<Pair> bt) {
        HashMap<Pair, String> charsMap = new HashMap<>();
        findEncoding(bt, charsMap, "");
        return charsMap;
    }

    // Recursive method to find the binary that represents a character.
    public static void findEncoding(BinaryTree<Pair> bt, HashMap<Pair, String> map, String prefix) {
        // If leaf node is found.
        if (bt.getLeft() == null && bt.getRight() == null) {
            map.put(bt.getData(), prefix); // Add the prefix to its corresponding pair.
        } else {
            findEncoding(bt.getLeft(), map, prefix + "0"); // Add 0 to prefix if going left.
            findEncoding(bt.getRight(), map, prefix + "1"); // Add 1 to prefix if going right.
        }
    }

    // Method to find the minimum pair between two queues.
    public static BinaryTree<Pair> findMin(ArrayList<BinaryTree<Pair>> q1, ArrayList<BinaryTree<Pair>> q2) {
        if (q1.isEmpty())
            return q2.remove(0);

        if (q2.isEmpty()) {
            return q1.remove(0);
        }

        if (q1.get(0).getData().compareTo(q2.get(0).getData()) < 0) {
            return q1.remove(0);
        }

        return q2.remove(0);
    }

    // Create Pairs by converting frequency into probability.
    public static ArrayList<Pair> getSortedPairs(HashMap<Character, Integer> map, int charLength) {
        ArrayList<Pair> pairList = new ArrayList<>();

        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            pairList.add(new Pair(entry.getKey(), Math.round(entry.getValue() * 10000d / charLength) / 10000d));
        }

        Collections.sort(pairList);
        return pairList; // Return a list of pairs with their respective probabilites.
    }

    // Map how often a character shows up to its respective character.
    public static HashMap<Character, Integer> mapFreq(char[] textChars) {
        HashMap<Character, Integer> map = new HashMap<>();

        for (char c : textChars) {
            if (!map.containsKey(c)) {
                map.put(c, 1); // Add new character to map.
            } else {
                map.put(c, map.get(c) + 1); // Increment count.
            }
        }

        return map;
    }

    // Get console reader.
    public static Scanner getConsole() {
        return new Scanner(System.in);
    }

    // Read text from file.
    public static String readText() throws IOException {
        Scanner fileReader = new Scanner(new File(getConsole().nextLine()));

        String text = "";

        while (fileReader.hasNext()) {

            text += fileReader.nextLine();

            if (fileReader.hasNextLine()) {
                text += "\n";
            }
        }

        fileReader.close();

        return text; // Return string from file.
    }

}
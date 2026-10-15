// Imports.
import java.io.IOException;

/**
 * The HuffmanDemo programs tests the Huffman program and its main methods.
 * 
 * @author Urmzd Mukhammadnaim.
 */
public class HuffmanDemo{

    public static void main(String[] args) throws IOException{

        // Message to let user know how to enter file names.
        System.out.println("Inorder, enter message file, encoded message file and finally, its dictionary file.");

        // Use methods.
        Huffman.encode();
        Huffman.decode();

        // Output that the file has been encoded and decoded successfully.
        System.out.println("The file inputted has successfully been encoded and decoded. Huffman.txt contains a table with the Huffman codes.");
    }

}
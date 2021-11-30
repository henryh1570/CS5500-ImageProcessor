
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashSet;

public class Test {

    public static void main(String[] args) {

        System.out.println("testing");
        
        int n = 512;
        ArrayList<String> list = new ArrayList<>();
        
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                int xp = (x+y) % n;
                int yp = (x+2*y) % n;
                String str = (""+xp+","+yp);
                
                if (list.contains(str)) {
                    System.out.println("CLONE: " + str);
                    System.exit(0);
                } else {
                    list.add(str);
                    System.out.println(x);
                }
            }
        }
        
        Collections.sort(list);
        
        for (String str: list) {
            System.out.println(str);
        }
        
//        int val = 8;
//        int pos = 3;
        
//        System.out.println("the bit of 8 at pos 3 is : " + ((8 >> 3)&1));
        /*
        try {
            File f = new File("codedfile.txt");
            FileOutputStream fos = new FileOutputStream(f);
            fos.write((char)256);
            fos.write((char)257);
            fos.close();
            FileInputStream fis = new FileInputStream(f);
            while(fis.available() > 0) {
                System.out.println(fis.read());
            }
            
        } catch(Exception e) {            
        }
        /*
        int count = 256;
        int multiple = count / 255;
        int remainder = count % 255;

        try {
            File f = new File("codedfile.txt");
            FileOutputStream fos = new FileOutputStream(f);
            BitSet bs = new BitSet();
            
            System.out.println("! is : " + (int)'!');
            fos.write('!');
        
            fos.flush();
            fos.close();

            FileInputStream fis = new FileInputStream("codedfile.txt");
            int sum = 0;
            while (fis.available() > 0) {
                int msg = fis.read();
                sum += msg;
                System.out.println(msg);
            }
            System.out.println("TOTAL: "+sum);

        } catch (Exception e) {
            System.err.println(e + " was found: Terminating");
        }
*/
    }

    //1-2 Partiton
    public static boolean canPart(int[] set) {
        int sum = 0;
        int n = set.length;

        for (int i = 0; i < n; i++) {
            sum += set[i];
        }

        if (sum % 3 != 0) {
            return false;
        }

        boolean[][] dp = new boolean[n + 1][sum / 3 + 1];

        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j <= sum / 3; ++j) {
                if (i == 0 || j == 0) {
                    dp[i][j] = false;
                } else if (set[i - 1] > j) {
                    dp[i][j] = dp[i - 1][j];
                } else if (set[i - 1] == j) {
                    dp[i][j] = true;
                } else {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - set[i - 1]];
                }
            }
        }

        for (int a = 0; a < dp.length; a++) {
            for (int b = 0; b < dp[a].length; b++) {
                String truth = "f";
                if (dp[a][b] == true) {
                    truth = "t";
                }
                System.out.print("[" + truth + "]");
            }
            System.out.println();
        }

        return dp[n][sum / 3];
    }
}

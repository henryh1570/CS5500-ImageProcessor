package com.mycompany.imageprocessor;

import java.awt.GraphicsEnvironment;
import java.awt.Point;
import javax.swing.JFrame;

/**
 * This program contains a suite of image processing algorithms with
 * a simple GUI. The App class is the main driver for the program; the GUI is
 * initialized here.
 * @author hh 
 */
public class App {

    static final String TEMP_FILENAME = "temp%img.bmp";
    static final String TEST_FILENAME = "ruler.tiff";
    
    public static void main(String[] args) {

        //Run the GUI MainScreen
        boolean GUIoff = true;
        if (GUIoff) {
            Processor p = new Processor();
            p.loadImageGrayscale(TEST_FILENAME);
            p.compressRLEGV("compressedfile");
            p.decompressRLEGV("compressedfile");
            try{
                Thread.sleep(3000);
            }catch(Exception e) {
                
            }
            p.saveImage("easy.tiff");
        } else {
            try {
                MainScreen screen = new MainScreen();
                Point center = GraphicsEnvironment.getLocalGraphicsEnvironment().getCenterPoint();
                screen.setSize(1200, 700);
                screen.setBounds(center.x - 1200 / 2, center.y - 700 / 2, 1200, 700);
                screen.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                screen.setVisible(true);
            } catch (Exception e) {
                System.err.println(e + " was found\nTerminating application");
                System.exit(0);
            }
        }
    }
}

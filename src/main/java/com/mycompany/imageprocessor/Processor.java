package com.mycompany.imageprocessor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import org.opencv.core.Core;
import static org.opencv.core.CvType.CV_8UC1;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * The processor class takes image inputs, applies a user selected algorithm,
 * and returns the newly created image.
 */
public class Processor {

    private Mat matrix;
    private Mat outputMatrix;
    private final Imgcodecs IMAGE_CODECS;
    private String outputName = "output";
    private String absoluteOutputFilePath = "";

    //Required library intializations for image loading.
    public Processor() {
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        IMAGE_CODECS = new Imgcodecs();
    }

    public void setOutputName(String name, String path) {
        outputName = name;
        absoluteOutputFilePath = path;
    }

    public String getOutputFilePath() {
        return absoluteOutputFilePath;
    }

    public String getOutputName() {
        return outputName;
    }

    public boolean loadImageGrayscale(String fileName) {
        try {
            //Specify file path and channel type
            matrix = IMAGE_CODECS.imread(fileName, Imgcodecs.IMREAD_GRAYSCALE);
        } catch (Exception e) {
            System.err.println(e + " failed to load file.");
            return false;
        }
        return true;
    }

    //8-bit Image Only for now.
    public void histogramEqualizationGlobal() {
        outputMatrix = new Mat(matrix.rows(), matrix.cols(), CV_8UC1);
        //Graylevels Rk are the keys. Values are the pixel coordinates. Values.size = Nk
        HashMap<Integer, ArrayList<String>> histogram = new HashMap<>();
        //Initialize the arraylists
        for (int n = 0; n < 256; n++) {
            histogram.put(n, new ArrayList<>());
        }

        //Create the histogram for the current image.
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                int grayValue = (int) matrix.get(i, j)[0];
                //Store pixel coordinates as string delimit via ","
                histogram.get(grayValue).add(i + "," + j);
            }
        }

        //Ratio = (L-1)/(M*N)
        final double RATIO = (histogram.keySet().size() - 1) / ((double) (matrix.rows() * matrix.cols()));
        ArrayList<Integer> equalizedHistogram = new ArrayList<>(Collections.nCopies(256, 0));
        int rKSum = 0;

        //Iterating rk=0 to 256, calculate the S-term
        for (int k = 0; k < histogram.keySet().size(); k++) {
            rKSum += histogram.get(k).size();
            int sValue = (int) Math.round(RATIO * rKSum);
            equalizedHistogram.set(k, sValue);
        }

        //Iterating rk=0 to 256, assign rk its new corresponding value, S-term.
        for (int a = 0; a < histogram.keySet().size(); a++) {
            //Retrieve all the coordinates of pixels of gray value rk = a
            ArrayList<String> list = histogram.get(a);
            int sTerm = equalizedHistogram.get(a);

            for (int b = 0; b < list.size(); b++) {
                String[] coordinates = list.get(b).split(",");
                int x = Integer.parseInt(coordinates[0]);
                int y = Integer.parseInt(coordinates[1]);

                outputMatrix.put(x, y, sTerm);
            }
        }

    }

    //Default mask: 3x3. User can define mask for NxN, N is odd and >3.
    //Let: center=(x,y). Then Start=(x-flr(n/2),y-flr(n/2)).
    public void histogramEqualizationLocal(int size) {
        outputMatrix = new Mat(matrix.rows(), matrix.cols(), CV_8UC1);
        int n = size;
        if (n < 3 || n % 2 == 0 || n > matrix.rows() || n > matrix.cols()) {
            n = 3;
        }
        final double RATIO = 255.0 / (double) (n * n);

        //Iterate the entire matrix
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                //Calculate the histogram for current center pixel (i,j)
                ArrayList<Integer> histogram = new ArrayList<>(Collections.nCopies(256, 0));
                //Offset from center = +/- flr(n/2)
                for (int a = (i - (n / 2)); a < (1 + i + (n / 2)); a++) {
                    for (int b = (j - (n / 2)); b < (1 + j + (n / 2)); b++) {
                        //Zero pad for out of bounds
                        int grayValue = 0;
                        if (a >= 0 && b >= 0 && a < matrix.rows() && b < matrix.cols()) {
                            grayValue = (int) matrix.get(a, b)[0];
                        }
                        histogram.set(grayValue, histogram.get(grayValue) + 1);
                    }
                }

                //Now calculate the s-terms for LHE
                int rKSum = 0;
                int centerValue = (int) matrix.get(i, j)[0];
                for (int c = 0; c < histogram.size(); c++) {
                    rKSum += histogram.get(c);
                    // Can terminate early if calculating center
                    if (c == centerValue) {
                        outputMatrix.put(i, j, Math.round(rKSum * RATIO));
                        c = histogram.size();
                    }
                }
            }
        }
    }

    /**
     * User chooses an NxN mask size that is odd and >= 3. Removes impulse
     * noise, but may produce blurring. Not allowing 0 padding for out of
     * bounds.
     */
    public void medianFilter(int size) {
        outputMatrix = new Mat(matrix.rows(), matrix.cols(), CV_8UC1);
        int n = size;
        if (n < 3 || n % 2 == 0 || n > matrix.rows() || n > matrix.cols()) {
            n = 3;
        }

        //Iterate the entire matrix
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {

                ArrayList<Integer> list = new ArrayList<>();
                int count = 0;
                //Iterate the current pixel's neighbors
                for (int a = (i - (n / 2)); a < (1 + i + (n / 2)); a++) {
                    for (int b = (j - (n / 2)); b < (1 + j + (n / 2)); b++) {
                        if (a >= 0 && b >= 0 && a < matrix.rows() && b < matrix.cols()) {
                            list.add((int) matrix.get(a, b)[0]);
                            count++;
                        }
                    }
                }
                //Replace current value with median value
                Collections.sort(list);
                outputMatrix.put(i, j, list.get((1 + count) / 2));
            }
        }
    }

    //TODO: Show the removal of lower and higher bit planes, their histograms, and their effects.
    //Remove certain bitplanes from 7 <--> 0, where 7 is MSB in an 8-bit image.
    public void removeBitplane(int[] bitPlanes) {
        outputMatrix = matrix.clone();
        //255 in binary
        int mask = 0b11111111;
        //Strip the mask of the specified bit positions
        for (Integer position : bitPlanes) {
            mask = mask & ~(1 << position);
        }

        //Bitwise AND the mask to every pixel value, stripping the bit-planes
        for (int i = 0; i < outputMatrix.rows(); i++) {
            for (int j = 0; j < outputMatrix.cols(); j++) {
                int value = (int) outputMatrix.get(i, j)[0];
                value = value & mask;
                outputMatrix.put(i, j, value);
            }
        }
    }

    /**
     * Currently saves an image without a specific level of compression. May
     * yield a different file size than expected.
     */
    public boolean saveImage() {
        try {
            IMAGE_CODECS.imwrite(absoluteOutputFilePath, outputMatrix);
            outputMatrix = null;
        } catch (Exception e) {
            System.err.println(e + " failed to save file.");
            outputMatrix = null;
            return false;
        }
        return true;
    }

    /**
     * Create a new desiredRows by desiredCols resolution image of the original
     * image. Alternating rows and cols are used to extract the value of pixels.
     */
    public void downscale(int desiredRows, int desiredCols) {
        outputMatrix = new Mat(desiredRows, desiredCols, CV_8UC1);
        // Find the scale factor for iterating the original image
        final double FACTOR_X = (matrix.rows() / (double) desiredRows);
        final double FACTOR_Y = (matrix.cols() / (double) desiredCols);

        //i iterates through cols, j iterates through rows
        for (int i = 0; i < desiredRows; i++) {
            for (int j = 0; j < desiredCols; j++) {
                //Find nearest corresponding pixel in the original matrix.
                double value = matrix.get((int) (j * FACTOR_X), (int) (i * FACTOR_Y))[0];
                outputMatrix.put(j, i, value);
            }
        }
    }

    /**
     * Create an upscale desiredRows by desiredCols image using 4
     * nearest-neighbors via euclidean distances.
     */
    public void zoomNearestNeighbor(int desiredRows, int desiredCols) {
        outputMatrix = new Mat(desiredRows, desiredCols, CV_8UC1);
        final double FACTOR_X = desiredRows / (double) matrix.rows();
        final double FACTOR_Y = desiredCols / (double) matrix.cols();

        for (int i = 0; i < desiredRows; i++) {
            double cy = (double) i / FACTOR_Y; //ConvertedY

            for (int j = 0; j < desiredCols; j++) {
                double cx = (double) j / FACTOR_X; //ConvertedX
                //Get the 4 nearest neighbors and find the euclidean distances.
                //Top Left
                int x2 = (int) cx;
                int y2 = (int) cy;
                int nearestX = x2;
                int nearestY = y2;
                double distance = getDistance(cx, cy, x2, y2);
                //Bottom Left
                x2 = (int) (j / FACTOR_X);
                y2 = (int) Math.ceil(i / FACTOR_Y);
                double distance2 = getDistance(cx, cy, x2, y2);
                //Check if the next y2 is not outOfBounds and which distance is bigger
                if (distance < distance2 && y2 < matrix.cols()) {
                    distance = distance2;
                    nearestX = x2;
                    nearestY = y2;
                }
                //Top Right
                x2 = (int) (j / FACTOR_X);
                y2 = (int) Math.floor(i / FACTOR_Y);
                distance2 = getDistance(cx, cy, x2, y2);
                //Check if the next x2 is not outOfBounds and which distance is bigger
                if (distance < distance2 && x2 < matrix.rows()) {
                    distance = distance2;
                    nearestX = x2;
                    nearestY = y2;
                }
                //Bottom Right
                x2 = (int) Math.ceil(j / FACTOR_X);
                y2 = (int) Math.ceil(i / FACTOR_Y);
                distance2 = getDistance(cx, cy, x2, y2);
                //Check if the next x2 and y2 is not outOfBounds and which distance is bigger
                if (distance < distance2 && x2 < matrix.rows() && y2 < matrix.cols()) {
                    nearestX = x2;
                    nearestY = y2;
                }
                outputMatrix.put(j, i, matrix.get(nearestX, nearestY)[0]);
            }
        }
    }

    private double getDistance(double x1, double y1, double x2, double y2) {
        return Math.sqrt(Math.pow((x2 - x1), 2) + Math.pow((y2 - y1), 2));
    }

    /**
     * To find the value for a pixel in the new, larger image, a conversion is
     * done to find it's approximate neighbors in the original image, in the X
     * direction (left to right, top to bottom).
     */
    public void zoomLinearX(int desiredRows, int desiredCols) {
        outputMatrix = new Mat(desiredRows, desiredCols, CV_8UC1);
        // Scale factor for the new:old image ratio.
        final double FACTOR_ROWS = desiredRows / (double) matrix.rows();
        final double FACTOR_COLS = desiredCols / (double) matrix.cols();

        //i iterates the y-axis, j iterates the x-axis
        for (int i = 0; i < desiredRows; i++) {
            double cy = (double) i / FACTOR_COLS; //ConvertedY

            for (int j = 0; j < desiredCols; j++) {
                double cx = (double) j / FACTOR_ROWS; //ConvertedX
                //Left and Right are the nearest-neighbor points
                int leftNeighbor = (int) cx;
                int rightNeighbor = (int) Math.ceil(cx);
                double leftValue = matrix.get(leftNeighbor, (int) cy)[0];
                double rightValue;
                double val;

                // If right neighbor is outOfBounds set its val as the boarder pixel.
                if (rightNeighbor < matrix.rows()) {
                    rightValue = matrix.get(rightNeighbor, (int) cy)[0];
                } else {
                    rightValue = matrix.get(matrix.rows() - 1, (int) cy)[0];
                }

                //Handle the case where Left and Right neighbors are the same.
                if (leftNeighbor != rightNeighbor) {
                    val = leftValue + (cx - leftNeighbor) * ((rightValue - leftValue) / (double) (rightNeighbor - leftNeighbor));
                } else {
                    val = leftValue;
                }
                //j = x-axis, i = y-axis
                outputMatrix.put(j, i, val);
            }
        }
    }

    /**
     * Linear interpolation upscale in the Y direction. We still iterate through
     * the image from left to right, top to bottom. Except that we approximate
     * the value of a new pixel via its top and bottom neighbors.
     */
    public void zoomLinearY(int desiredRows, int desiredCols) {
        outputMatrix = new Mat(desiredRows, desiredCols, CV_8UC1);
        // Scale factor for the new:old image ratio.
        final double FACTOR_ROWS = desiredRows / (double) matrix.rows();
        final double FACTOR_COLS = desiredCols / (double) matrix.cols();
        //i iterates the y-axis, j iterates the x-axis
        for (int i = 0; i < desiredRows; i++) {

            double cy = (double) i / FACTOR_COLS; //ConvertedY
            int topNeighbor = (int) cy;
            int bottomNeighbor = (int) Math.ceil(cy);

            for (int j = 0; j < desiredCols; j++) {
                double cx = (double) j / FACTOR_ROWS; //ConvertedX
                //top and bottom are the nearest-neighbor points

                double topValue = matrix.get((int) cx, topNeighbor)[0];
                double bottomValue;
                double val;

                // If bottom neighbor is outOfBounds set its val as the boarder pixel.
                if (bottomNeighbor < matrix.cols()) {
                    bottomValue = matrix.get((int) cx, bottomNeighbor)[0];
                } else {
                    bottomValue = matrix.get((int) cx, matrix.cols() - 1)[0];
                }

                //Handle the case where top and bottom neighbors are the same.
                if (topNeighbor != bottomNeighbor) {
                    val = topValue + (cy - topNeighbor) * ((bottomValue - topValue) / (double) (bottomNeighbor - topNeighbor));
                } else {
                    val = topValue;
                }
                //j = x-axis, i = y-axis
                outputMatrix.put(j, i, val);
            }
        }
    }

    /**
     * Combining linear interpolation in the x and y direction to produce an
     * even more accurate upscale image. 4 Points are selected to use given a
     * converted point we are trying to find the value for.
     */
    public void zoomBilinear(int desiredRows, int desiredCols) {
        outputMatrix = new Mat(desiredRows, desiredCols, CV_8UC1);
        // Scale factor for the new:old image ratio.
        final double FACTOR_ROWS = desiredRows / (double) matrix.rows();
        final double FACTOR_COLS = desiredCols / (double) matrix.cols();

        //i iterates the y-axis, j iterates the x-axis
        for (int i = 0; i < desiredRows; i++) {
            double cy = (double) i / FACTOR_COLS;       //ConvertedY

            for (int j = 0; j < desiredCols; j++) {
                double cx = (double) j / FACTOR_ROWS;   // ConvertedX
                double val = 0;

                //There are 4 points needed, and various cases.
                //On the grid, X is increasing left to right
                //Y is increasing top to bottom
                int x1 = (int) cx;
                int x2 = (int) Math.ceil(cx);
                int y1 = (int) Math.ceil(cy);
                int y2 = (int) cy;

                //Case 1: check if the Point is already existing in the original image.
                //Case 2: The neighbors are all equal
                if ((cx % 1 == 0 && cy % 1 == 0) || (x1 == x2 && x1 == y1 && y1 == y2)) {
                    val = matrix.get((int) cx, (int) cy)[0];
                } else {
                    //Check for outOfBounds on the ceiling variables.
                    if (x2 >= matrix.rows()) {
                        x2 = matrix.rows() - 1;
                    }
                    if (y1 >= matrix.cols()) {
                        y1 = matrix.cols() - 1;
                    }

                    double q12 = matrix.get(x1, y2)[0]; //TopLeft value
                    double q22 = matrix.get(x2, y2)[0]; //TopRight value
                    double q11 = matrix.get(x1, y1)[0]; //BottomLeft value
                    double q21 = matrix.get(x2, y1)[0]; //BottomRight value

                    //Case 3: Use linearY since there is no unique x-cooridnate
                    if (x1 == x2) {
                        double topNeighbor = y2;
                        double bottomNeighbor = y1;
                        double topValue = q12;
                        double bottomValue = q11;
                        val = topValue + (cy - topNeighbor) * ((bottomValue - topValue) / (double) (bottomNeighbor - topNeighbor));
                        //Case 4: Use linearX since there is no unique y-coordinate
                    } else if (y1 == y2) {
                        double leftNeighbor = x1;
                        double rightNeighbor = x2;
                        double leftValue = q11;
                        double rightValue = q21;
                        val = leftValue + (cx - leftNeighbor) * ((rightValue - leftValue) / (double) (rightNeighbor - leftNeighbor));
                        //Case 5: Use Bilinear since there are no conflicts
                    } else {
                        double denominator = ((x2 - x1) * (y2 - y1));
                        val += (q11 * (x2 - cx) * (y2 - cy));
                        val += (q21 * (cx - x1) * (y2 - cy));
                        val += (q12 * (x2 - cx) * (cy - y1));
                        val += (q22 * (cx - x1) * (cy - y1));
                        val = (val / denominator);
                    }
                }
                //j = x-axis, i = y-axis
                outputMatrix.put(j, i, val);
            }
        }
    }

    /**
     * Because image bit depth cannot be reduced by library, the values will be
     * altered proportionally to the bit depth. 0 = black. currentDepth-1 =
     * white Total Equation: Floor(PropConversion * MaxRangeRatio) Total
     * Equation: Floor(Floor(val/(2^c-t)) * ((2^c-1)/(2^t-1))
     */
    public void reduceGraylevel(int targetDepth) {
        //Create new matrix with specified rows, cols, bit-depth, and channels
        outputMatrix = new Mat(matrix.rows(), matrix.cols(), CV_8UC1);
        int currentDepth = 8;
        double depthRatio = Math.pow(2, currentDepth - targetDepth);

        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                //pixel value is stored as a single element double[]
                double val = matrix.get(i, j)[0];
                double proportionalConversion = Math.floor(val / depthRatio);
                double maxRangeRatio = ((Math.pow(2, currentDepth) - 1) / ((Math.pow(2, targetDepth)) - 1));
                double newVal = (Math.floor(proportionalConversion * maxRangeRatio));
                //Put the new value back into the matrix
                outputMatrix.put(i, j, newVal);
            }
        }
    }

    //Helper method for laplacian and smooth filters.
    private void convolution(double[][] mask) {
        outputMatrix = new Mat(matrix.rows(), matrix.cols(), CV_8UC1);

        int n = mask.length;
        //Iterate the entire matrix
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {

                double sum = 0;
                //Iterate the neighborhood of the current pixel
                for (int a = (i - (n / 2)); a < (1 + i + (n / 2)); a++) {
                    for (int b = (j - (n / 2)); b < (1 + j + (n / 2)); b++) {
                        //Ignore the outofbounds
                        if (a >= 0 && b >= 0 && a < matrix.rows() && b < matrix.cols()) {
                            int posX = a - (i - (n/2));
                            int posY = b - (j - (n/2));
                            sum += (matrix.get(a, b)[0] * mask[posX][posY]);
                        }
                    }
                }
                outputMatrix.put(i, j, (int) sum);
            }
        }

    }
    
    /**
     * High boost filter allows user to define the smoothing operation to use,
     * including mask size and type, then can decide its weight to be added to
     * the original image, resulting in the unsharp masking.
     */
    public void highBoostingFilter(int size, double weight, String filterType) {
        double k = weight;
        //temporary outputMatrix is generated
        smoothingFilter(size, filterType); 
        Mat blurredMatrix = outputMatrix.clone();
        
        //Highboost = original - k*(original - blurred)
        for (int i = 0; i < blurredMatrix.rows(); i++) {
            for (int j = 0; j < blurredMatrix.cols(); j++) {
                double originalValue = matrix.get(i,j)[0];
                double blurredValue = blurredMatrix.get(i, j)[0];
                double value = originalValue + (k*(originalValue - blurredValue));
                outputMatrix.put(i, j, (int)value);
            }
        }
        blurredMatrix = null;
    }
    
    /**
     * This Laplacian Operator produces kernels of only 1s surrounding the 
     * larger center value, of which the numbers can all be flipped neg/pos.
     * TODO: Diagonal and other variants of Laplacian may be implemented later.
     */
    public void sharpeningLaplacianFilter(int size, boolean centerIsPositive, boolean diagonalsIncluded) {
        int n = size;
        if (n < 3 || n % 2 == 0 || n > matrix.rows() || n > matrix.cols()) {
            n = 3;
        }
        
        int sign = 1;
        if (!centerIsPositive) {
            sign = -sign;
        }
        double[][] mask = new double[n][n];
        
        //Including Diagonals
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                //Make non-center = +/- 1
                if (!((i==n/2) && (j==n/2))) {
                    mask[i][j] = -1 * sign;
                } else {
                    mask[i][j] = ((n*n) - 1) * sign;
                }
            }
        }
        convolution(mask);
    }
    
    /**
     * The user can choose to use Gaussian, weighted average, or box filter
     * kernels to create a blurring effect on the image. User can choose kernel
     * size as well.
     * TODO: Allow user to change sigma, sd.
     */
    public void smoothingFilter(int size, String type) {
        int n = size;
        if (n < 3 || n % 2 == 0 || n > matrix.rows() || n > matrix.cols()) {
            n = 3;
        }
        
        double total = 0;
        
        double[][] mask = new double[n][n];
        switch (type) {
            case "Gaussian":                                
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        //Gaussian Formula for 2d, x2 and y2 are dist from cent.
                        double sd = 1.0;
                        int x2 = (n/2 - i) * (n/2 -i);
                        int y2 = (n/2 - j) * (n/2 -j);
                        double numerator = Math.pow(Math.E, -1*(x2+y2)/(2*sd*sd));
                        double denominator = (2*Math.PI*sd*sd);
                        //value is the normalized ratio of pixel val/h(s,t)
                        double value = numerator/denominator;
                        total += value;
                        mask[i][j] = value;
                    }
                }
                // Redistribute remainder of sum to make sure h(s,t) = 1
                for (int a = 0; a < n; a++) {
                    for (int b = 0; b < n; b++) {
                        //Add the remainder to the pixel using its ratio
                        mask[a][b] += mask[a][b]*(1/total)*((1-total));
                    }
                }
                break;
            case "Weighted":                
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double value = 0;
                        //Assing value based on city-block distance
                        if (!(i==n/2 && j==n/2)) {
                            int cityBlockDist = Math.abs(n/2 - i) + Math.abs(n/2 - j);
                            value = Math.pow(2, n - 1 -cityBlockDist);
                        } else {
                            //Center value: the highest power of 2.
                            value = Math.pow(2, n - 1);
                        }
                        total += value;
                        mask[i][j] = value;
                    }
                }
                // Inefficient reweighting of the kernel. Makes sure h(s,t) = 1
                for (int a = 0; a < n; a++) {
                    for (int b = 0; b < n; b++) {
                        mask[a][b] = mask[a][b] * (1/total);
                    }
                }
                break;
            default: //"Box"
               for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        mask[i][j] = 1/(double)(n*n);
                    }
                }                
                break;                
        }
        convolution(mask);
    }
}

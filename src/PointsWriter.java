import java.io.FileWriter;
import java.io.IOException;

// this class can write points to a file
public class PointsWriter {
    private Double[] x;
    private Double[] y;
    private int size;
    private String fileName;
    private String filePath;
    public PointsWriter(Double[] x, Double[] y, int size, String fileName) {
        this.x = x;
        this.y = y;
        this.size = size;
        this.fileName = fileName;
        this.filePath = "src/Points/" + fileName + ".txt";
    }
    public void writePoints() {
        try {
            FileWriter fw = new FileWriter(filePath);
            for (int i = 0; i < size; i++) {
                fw.write(x[i] + " " + y[i] + "\n");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void plotPoints() {
        String command = "python3 src/plotPoints.py";
        try {
            Process p = Runtime.getRuntime().exec(command);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        PointsWriter pointsWriter = new PointsWriter(new Double[]{}, new Double[]{}, 5, "plotPoints");
        pointsWriter.plotPoints();
    }
}

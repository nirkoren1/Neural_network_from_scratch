import java.io.BufferedWriter;
import java.io.File;
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
        this.filePath = "src/Points2/" + fileName + ".txt";
    }
    public void writePoints() {
        try {
            File file = new File(filePath);
            if (!file.exists()) {
                file.createNewFile();
            }
            FileWriter fw = new FileWriter(file.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);
            for (int i = 0; i < size; i++) {
                bw.write(x[i] + " " + y[i] + "\n");
            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

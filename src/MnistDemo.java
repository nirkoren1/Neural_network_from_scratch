import Functions.*;
import Matrix.Matrix;

import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.io.File;
import java.util.Scanner;



public class MnistDemo {
    public static double[] predict(Model model, Matrix x) {
        Matrix logits = model.feedForward(x);
        double[] out = new double[logits.getRows()];
        for (int i = 0; i < logits.getRows(); i++) {
            out[i] = logits.getValues()[i][0];
        }
        return out;
//        int argMax = 0;
//        double max = -1;
//        for (int j = 0; j < logits.getRows(); j++) {
//            if (logits.getValues()[j][0] > max) {
//                max = logits.getValues()[j][0];
//                argMax = j;
//            }
//        }
//        return argMax;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        String pathToModel = args[0];
        String pathToResults = args[1];
        Model model = Model.loadModel(pathToModel);
        Matrix x = new Matrix(args.length - 2, 1);
        for (int i = 2; i < args.length; i++) {
            x.getValues()[i - 2][0] = Double.parseDouble(args[i]) / 255;
        }
        double[] y = predict(model, x);
        FileWriter myWriter = new FileWriter(pathToResults);
        StringBuilder st = new StringBuilder("");
        for (double v : y) {
            st.append(v).append(" ");
        }
        myWriter.write(st.toString());
        myWriter.close();
    }
}

import Functions.*;
import Matrix.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RegressionEx {
    private static Random random = new Random();

    public static void writePointFile(Model model, int index) {
        int testSize = 20000;
        Double[] all_x = new Double[testSize];
        Double[] all_y = new Double[testSize];
        for (int i = 0; i < testSize; i++) {
            double x = i / 1000.0 - 10.0;
            Matrix xMatrix = new Matrix(1, 1);
            xMatrix.getValues()[0][0] = x;
            double y = model.feedForward(xMatrix).getValues()[0][0];
            all_x[i] = x;
            all_y[i] = y;
        }
        PointsWriter pointsWriter = new PointsWriter(all_x, all_y,testSize,  "points_" + index);
        pointsWriter.writePoints();
    }

    public static void main(String[] args) {
        List<Matrix> X = new ArrayList<>();
        List<Matrix> Y = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            Matrix x = new Matrix(1, 1);
            for (int j = 0; j < 1; j++) {
                x.getValues()[j][0] = -16 + 32 * random.nextDouble();
            }
            X.add(x);
            Matrix y = new Matrix(1, 1);
            y.getValues()[0][0] = 2 * Math.pow(x.getValues()[0][0], 2) + 1;
            Y.add(y);
        }

        List<Matrix> XTest = new ArrayList<>();
        List<Matrix> YTest = new ArrayList<>();
        int testSize = 1000;
        for (int i = 0; i < testSize; i++) {
            Matrix x = new Matrix(1, 1);
            for (int j = 0; j < 1; j++) {
                x.getValues()[j][0] = -16 + 32 * random.nextDouble();
            }
            XTest.add(x);
            Matrix y = new Matrix(1, 1);
            y.getValues()[0][0] = 2 * Math.pow(x.getValues()[0][0], 2) + 1;
            YTest.add(y);
        }


        Model model = new Model(64, new MSE(), 0.00001);
        model.add(1, new None());
        model.add(16, new Relu());
        model.add(32, new Relu());
        model.add(32, new None());
        model.add(16, new None());
        model.add(1, new None());

        model.trainModel(X, Y, 30, XTest, YTest, 1);

        System.out.println("----------TEST-----------");

        double avgError = 0;
        for (int i = 0; i < testSize; i++) {
            double xPredict = model.feedForward(XTest.get(i)).getValues()[0][0];
            double yTrue = YTest.get(i).getValues()[0][0];
            avgError += Math.pow(xPredict - yTrue, 2);
            System.out.println(XTest.get(i));
            System.out.println(yTrue);
            System.out.println(xPredict);
            System.out.println("---------------------");
        }
        avgError /= testSize;
        System.out.println("RMSE error = " + Math.sqrt(avgError));
    }
}

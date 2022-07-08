import Functions.MSE;
import Functions.None;
import Functions.Relu;
import Functions.Sigmoid;
import Matrix.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RegressionEx {
    private static Random random = new Random();

    public static void writePointFile(Model model, int index) {
        int testSize = 10000;
        Double[] all_x = new Double[testSize];
        Double[] all_y = new Double[testSize];
        for (int i = 0; i < testSize; i++) {
            double x = i / 1000.0;
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
            y.getValues()[0][0] = 2 * x.getValues()[0][0] + 1;
            Y.add(y);
        }

        Model model = new Model(16, new MSE(), 0.0005);
        model.add(1, new Relu());
        model.add(16, new Sigmoid());
//        model.add(32, new Sigmoid());
//        model.add(32, new Sigmoid());
        model.add(16, new Sigmoid());
        model.add(1, new None());

        model.trainModel(X, Y, 200);

        System.out.println("----------TEST-----------");
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
            y.getValues()[0][0] = 2 * x.getValues()[0][0] + 1;
            YTest.add(y);
        }

        double avgError = 0;
        for (int i = 0; i < testSize; i++) {
            double xPredict = model.feedForward(XTest.get(i)).getValues()[0][0];
            double yTrue = YTest.get(i).getValues()[0][0];
            avgError += Math.abs(xPredict - yTrue);
            System.out.println(XTest.get(i));
            System.out.println(yTrue);
            System.out.println(xPredict);
            System.out.println("---------------------");
        }
        avgError /= testSize;
        System.out.println("Total error = " + avgError);
    }
}
